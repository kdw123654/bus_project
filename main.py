import os
import requests
from urllib.parse import quote
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("BUS_API_KEY")
ODSAY_API_KEY = os.getenv("ODSAY_API_KEY")

STATION_ID = "119000302"         # 사당역 9번 출구 앞 (승차)
ROUTE_REGULAR_ID = "200000149"    # 일반 7790
ROUTE_RESERVE_ID = "233000384"    # 7790 (예약/전세)

ODSAY_HEADERS = {'Referer': 'http://127.0.0.1:8000'}  # ODsay API 호출 시 Referer 헤더 필요

app = FastAPI(title="7790 & Smart Transit Router")

if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

CACHED_STATIONS = []


def fetch_full_route_stations():
    """GBIS에서 7790 전 노선 66개 정류장 수집"""
    url = "https://www.gbis.go.kr/gbis2014/schBusAPI.action"
    data = {"cmd": "searchRouteJson", "routeId": ROUTE_REGULAR_ID}
    try:
        r = requests.post(url, data=data, timeout=5, verify=False)
        res = r.json()
        up_list = res.get("result", {}).get("gg", {}).get("up", {}).get("list", [])
        down_list = res.get("result", {}).get("gg", {}).get("down", {}).get("list", [])

        stations = []
        for s in up_list:
            name = s.get("stationNm", "")
            is_pass = "(경유)" in name or "(미정차)" in name
            clean_name = name.replace("(경유)", "").replace("(미정차)", "")
            stations.append({
                "seq": int(s.get("stationSeq", 0)),
                "id": str(s.get("stationId")),
                "name": clean_name,
                "rawName": name,
                "isPass": is_pass,
                "dir": "UP",
                "isTurn": (name == "사당역(중)"),
                "isTarget": False
            })

        for s in down_list:
            name = s.get("stationNm", "")
            is_pass = "(경유)" in name or "(미정차)" in name
            clean_name = name.replace("(경유)", "").replace("(미정차)", "")
            stations.append({
                "seq": int(s.get("stationSeq", 0)),
                "id": str(s.get("stationId")),
                "name": clean_name,
                "rawName": name,
                "isPass": is_pass,
                "dir": "DOWN",
                "isTurn": False,
                "isTarget": (name == "사당역9번출구앞")
            })

        print(f"✅ 7790 전 노선 {len(stations)}개 정류장 로드 완료!")
        return stations
    except Exception as e:
        print(f"❌ 정류소 목록 로드 실패: {e}")
        return []


@app.on_event("startup")
def startup_event():
    global CACHED_STATIONS
    CACHED_STATIONS = fetch_full_route_stations()


@app.get("/")
def get_home():
    return FileResponse("static/index.html")


@app.get("/api/buses")
def get_bus_arrivals():
    global CACHED_STATIONS
    if not CACHED_STATIONS:
        CACHED_STATIONS = fetch_full_route_stations()

    url_arrival = f"https://apis.data.go.kr/6410000/busarrivalservice/v2/getBusArrivalListv2?serviceKey={API_KEY}&stationId={STATION_ID}&format=json"
    url_route = "https://www.gbis.go.kr/gbis2014/schBusAPI.action"
    data_route = {"cmd": "searchRouteJson", "routeId": ROUTE_REGULAR_ID}

    arriving_buses = []
    live_vehicles = []

    # 1. 사당역 도착 정보
    try:
        r_arr = requests.get(url_arrival, timeout=4)
        if r_arr.status_code == 200:
            arr_data = r_arr.json()
            bus_list = arr_data.get("response", {}).get("msgBody", {}).get("busArrivalList", [])
            for bus in bus_list:
                route_id = str(bus.get("routeId"))
                route_name = str(bus.get("routeName"))
                if route_id in [ROUTE_REGULAR_ID, ROUTE_RESERVE_ID] or "7790" in route_name:
                    is_resv = (route_id == ROUTE_RESERVE_ID) or ("예약" in route_name)
                    if bus.get("predictTime1"):
                        plate = bus.get("plateNo1", "")
                        arriving_buses.append({
                            "routeName": route_name,
                            "isReserved": is_resv,
                            "predictTimeMin": int(bus.get("predictTime1")),
                            "remainStations": int(bus.get("locationNo1") or 0),
                            "remainSeats": int(bus.get("remainSeatCnt1") or 0),
                            "plateNo": plate,
                            "shortPlate": plate[-4:] if len(plate) >= 4 else plate
                        })
                    if bus.get("predictTime2"):
                        plate = bus.get("plateNo2", "")
                        arriving_buses.append({
                            "routeName": route_name,
                            "isReserved": is_resv,
                            "predictTimeMin": int(bus.get("predictTime2")),
                            "remainStations": int(bus.get("locationNo2") or 0),
                            "remainSeats": int(bus.get("remainSeatCnt2") or 0),
                            "plateNo": plate,
                            "shortPlate": plate[-4:] if len(plate) >= 4 else plate
                        })
            arriving_buses.sort(key=lambda x: x["predictTimeMin"])
    except Exception as e:
        print(f"도착 API 오류: {e}")

    # 2. 노선 전체 실시간 차량 위치
    try:
        r_live = requests.post(url_route, data=data_route, timeout=4, verify=False)
        if r_live.status_code == 200:
            live_data = r_live.json()
            v_list = live_data.get("result", {}).get("realTime", {}).get("list", [])
            for v in v_list:
                plate_list = v.get("busNo", [])
                plate = plate_list[0] if plate_list else ""
                short_plate = plate[-4:] if len(plate) >= 4 else plate
                is_reserved = (v.get("reservation") == "Y")
                target_station = str(v.get("toStationId") or v.get("fromStationId") or "")
                
                live_vehicles.append({
                    "plateNo": plate,
                    "shortPlate": short_plate,
                    "remainSeats": int(v.get("seat") or 0),
                    "isReserved": is_reserved,
                    "isDoubleDecker": (v.get("lowPlate") == "2"),
                    "currentStationId": target_station,
                    "dir": v.get("busDirList", [""])[0]
                })
    except Exception as e:
        print(f"차량 API 오류: {e}")

    return {
        "success": True,
        "stations": CACHED_STATIONS,
        "arrivingBuses": arriving_buses,
        "vehicles": live_vehicles
    }



# ⭐ 3. 신규 기능: ODsay 기반 대중교통 최적 경로 탐색 엔드포인트
@app.get("/api/search_place")
def search_place(keyword: str = Query(..., description="검색할 장소명")):
    clean_kw = keyword.strip()
    if clean_kw in name or name in clean_kw:
        return {'success': False, 'places': []}

    results = []

    if ODSAY_API_KEY:
        url = "https://api.odsay.com/v1/api/searchStation"
        params = {"stationName": clean_kw, "apiKey": ODSAY_API_KEY}
        try:
            r = requests.get(url, params=params, headers=ODSAY_HEADERS, timeout=4)
            st_list = r.json().get("result", {}).get("station", [])
            for st in st_list[:5]:
                st_name = st.get("stationName", "")
                city = st.get("cityName", "") or st.get("do", "")
                results.append({
                    "name": st_name,
                    "detail": f"{city} 정류장/역",
                    "x": float(st.get("x")),
                    "y": float(st.get("y"))
                })
        except Exception as e:
            print(f"장소 검색 오류: {e}")

    try:
        url_nom = "https://nominatim.openstreetmap.org/search"
        headers_nom = {"User-Agent": "SuwonBusProject/1.0"}
        params_nom = {"q": clean_kw, "format": "json", "countrycodes": "kr", "limit": 4}
        r_nom = requests.get(url_nom, params=params_nom, headers=headers_nom, timeout=4)

        for item in r_nom.json():
            name = item.get("name") or clean_kw
            display = item.get("display_name", "").split(",")[0]
            results.append({
                "name": name,
                "detail": display,
                "x": float(item.get("lon")),
                "y": float(item.get("lat"))
            })
    except Exception as e:
        print(f"Nominatim 검색 오류: {e}")

    unique_results = []
    seen = set()
    for item in results:
        key = (round(item["x"], 3), round(item["y"], 3))
        if key not in seen:
            seen.add(key)
            unique_results.append(item)
    return {"success": True, "places": unique_results[:6]}

@app.get("/api/directions")
def get_transit_directions(
    startX: float = Query(..., description="출발지 경도(X)"),
    startY: float = Query(..., description="출발지 위도(Y)"),
    endX: float = Query(..., description="목적지 경도(X)"),
    endY: float = Query(..., description="목적지 위도(Y)")
):
    if not ODSAY_API_KEY:
        return {"success": False, "message": "ODSAY API Key가 설정되지 않았습니다."}
    dist_approx = ((startX - endX)**2 + (startY - endY)**2)**0.5
    if dist_approx < 0.005:
        return {"success": False, "error": "출발지와 목적지가 너무 가깝습니다 (약 500m 이내, 도보 이동 권장)."}
    url = f"https://api.odsay.com/v1/api/searchPubTransPathT?SX={startX}&SY={startY}&EX={endX}&EY={endY}&OPT=0&apiKey={quote(ODSAY_API_KEY)}"

    try:
        r = requests.get(url, headers=ODSAY_HEADERS, timeout=4)
        data = r.json()

        if 'error' in data:
          err_val = data['error']
          if isinstance(err_val, list) and len(err_val) > 0:
                err_msg = err_val[0].get("message") or err_val[0].get("msg") or "경로를 찾을 수 없습니다."
          elif isinstance(err_val, dict):
                err_msg = err_val.get("message") or err_val.get("msg") or "경로를 찾을 수 없습니다."
          else:
                err_msg = str(err_val)
          return {"success": False, "error": err_msg}

        paths = data.get("result", {}).get("path", [])
        parsed_paths = []
        first_7790 = None
        bus_check = get_bus_arrivals()
        if bus_check.get("arrivingBuses"):
            first_7790 = bus_check["arrivingBuses"][0]
        for p in paths[:3]:
            info = p.get("info", {})
            sub_paths = p.get("subPath", [])
            steps = []
            includes_7790 = False
            for sp in sub_paths:
                traffic_type = sp.get("trafficType")
                section_time = sp.get("sectionTime", 0)
                
                if traffic_type == 1:
                    subway_lanes = [lane.get("name") for lane in sp.get("lane", [])]
                    steps.append({
                        "type": "subway",
                        "name": ", ".join(subway_lanes),
                        "time": section_time,
                        "start": sp.get("startName"),
                        "end": sp.get("endName")
                    })
                elif traffic_type == 2:
                    bus_lanes = [lane.get("busNo") for lane in sp.get("lane", [])]
                    bus_name_str = ", ".join(bus_lanes)
                    if "7790" in bus_name_str:
                        includes_7790 = True
                        
                    steps.append({
                        "type": "bus",
                        "name": bus_name_str,
                        "time": section_time,
                        "start": sp.get("startName"),
                        "end": sp.get("endName")
                    })
                elif traffic_type == 3 and section_time > 0:
                    steps.append({
                        "type": "walk",
                        "time": section_time,
                        "distance": sp.get("distance", 0)
                    })
            parsed_paths.append({
                "pathType": p.get("pathType"),
                "totalTime": info.get("totalTime"),
                "payment": info.get("payment", 0),
                "transferCount": max(0, info.get("busTransitCount", 0) + info.get("subwayTransitCount", 0) - 1),
                "totalWalk": info.get("totalWalk", 0),
                "includes7790": includes_7790,
                "status7790": first_7790 if includes_7790 else None,
                "steps": steps
            })
        return {"success": True, "count": len(parsed_paths), "paths": parsed_paths}
    except Exception as e:
        return {"success": False, "error": str(e), "paths": []}
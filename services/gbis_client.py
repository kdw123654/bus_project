import time
import threading
import requests
from requests.adapters import HTTPAdapter
import urllib3
from typing import List, Dict, Any, Tuple
from core.config import (
    API_KEY,
    ROUTE_REGULAR_ID,
    ROUTE_RESERVE_ID,
    GBIS_HEADERS,
    GBIS_ROUTE_CACHE_TTL,
    TRAFFIC_CACHE_TTL,
    WEATHER_CACHE_TTL
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 공용 HTTP 커넥션 풀 세션 (TCP/TLS 핸드셰이크 재사용으로 레이턴시 단축)
HTTP_SESSION = requests.Session()
_adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20)
HTTP_SESSION.mount("https://", _adapter)
HTTP_SESSION.mount("http://", _adapter)
HTTP_SESSION.headers.update(GBIS_HEADERS)

# 인메모리 캐시 및 Cache Stampede 방지용 스레드 락
_CACHE_GBIS_ROUTE: Dict[str, Any] = {"data": None, "timestamp": 0.0}
_CACHE_TRAFFIC_DELAYS: Dict[str, Any] = {}
_CACHE_WEATHER: Dict[str, Any] = {"data": None, "timestamp": 0.0}
_LOCK_GBIS_ROUTE = threading.Lock()
_LOCK_TRAFFIC_DELAYS = threading.Lock()
_LOCK_WEATHER = threading.Lock()


def is_charter_or_reserved_bus(
    route_id: str = "",
    route_name: str = "",
    plate_no: str = "",
    short_plate: str = "",
    reservation_flag: str = ""
) -> Tuple[bool, str]:

    clean_plate = (plate_no or "").replace(" ", "")
    clean_short = (short_plate or "").replace(" ", "")
    if not clean_short and clean_plate:
        clean_short = clean_plate[-4:] if len(clean_plate) >= 4 else clean_plate


    if str(route_id) == ROUTE_RESERVE_ID or ("예약" in str(route_name)):
        return True, "GBIS 공공데이터 예약 노선(233000384)으로 지정됨"


    if str(reservation_flag).upper() == "Y":
        return True, "GBIS 실시간 차량 예약(Reservation) 플래그 식별됨"


    charter_plate_markers = ["경기79", "79사", "79바", "79아", "79자", "경기78", "78사", "78바", "78아", "78자"]
    for marker in charter_plate_markers:
        if marker in clean_plate:
            return True, f"전세버스 면허('{marker}'): 외부 임차 전세 차량으로 식별됨 (정규 70사/76아 외 면허)"
    if any(m in clean_plate for m in ["서울70사", "서울70바", "서울70아", "서울70자"]):
        return True, f"외부 전세버스 면허('{clean_plate}'): 타 시도 임차 전세 차량"

    if clean_short.startswith("9"):
        return True, f"차량번호 '{clean_short}': 9000번대 외부 임차 전세버스 식별 번호"

    return False, "정규 광역버스 면허 (경기 70사 수원 / 경기 76아 화성)"


def fetch_gbis_route_cached(force_refresh: bool = False) -> Dict[str, Any]:
    global _CACHE_GBIS_ROUTE
    now = time.time()
    if not force_refresh and _CACHE_GBIS_ROUTE["data"] and (now - _CACHE_GBIS_ROUTE["timestamp"] < GBIS_ROUTE_CACHE_TTL):
        return _CACHE_GBIS_ROUTE["data"]

    with _LOCK_GBIS_ROUTE:
        now = time.time()
        if not force_refresh and _CACHE_GBIS_ROUTE["data"] and (now - _CACHE_GBIS_ROUTE["timestamp"] < GBIS_ROUTE_CACHE_TTL):
            return _CACHE_GBIS_ROUTE["data"]

        url_route = "https://www.gbis.go.kr/gbis2014/schBusAPI.action"
        data_route = {"cmd": "searchRouteJson", "routeId": ROUTE_REGULAR_ID}
        try:
            try:
                r_live = HTTP_SESSION.post(url_route, data=data_route, timeout=5, verify=True)
            except requests.exceptions.SSLError:
                r_live = HTTP_SESSION.post(url_route, data=data_route, timeout=5, verify=False)
            if r_live.status_code == 200:
                res = r_live.json()
                _CACHE_GBIS_ROUTE = {"data": res, "timestamp": time.time()}
                return res
        except Exception as e:
            print(f"GBIS 노선 통합 캐시 조회 오류: {e}")

        return _CACHE_GBIS_ROUTE["data"] or {}


def parse_gbis_realtime_vehicles(live_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """GBIS 실시간 노선 데이터에서 운행 차량 리스트를 일관된 포맷으로 파싱 (DRY 단일 함수)"""
    vehicles: List[Dict[str, Any]] = []
    if not live_data:
        return vehicles

    v_list = live_data.get("result", {}).get("realTime", {}).get("list", [])
    for v in v_list:
        plate_list = v.get("busNo", [])
        plate = plate_list[0] if plate_list else ""
        short_plate = plate[-4:] if len(plate) >= 4 else plate
        is_reserved, resv_reason = is_charter_or_reserved_bus(
            route_id=ROUTE_REGULAR_ID,
            route_name="7790",
            plate_no=plate,
            short_plate=short_plate,
            reservation_flag=v.get("reservation", "")
        )
        from_st = str(v.get("fromStationId") or "")
        to_st = str(v.get("toStationId") or "")
        pos_raw = v.get("busPosition")
        bus_pos = ""
        if isinstance(pos_raw, list) and pos_raw:
            bus_pos = str(pos_raw[0]).strip()
        elif isinstance(pos_raw, str):
            bus_pos = pos_raw.strip()

        target_station = from_st or to_st

        vehicles.append({
            "plateNo": plate,
            "shortPlate": short_plate,
            "remainSeats": int(v.get("seat") or 0),
            "isReserved": is_reserved,
            "reserveReason": resv_reason,
            "isDoubleDecker": (v.get("lowPlate") == "2"),
            "currentStationId": target_station,
            "fromStationId": from_st,
            "toStationId": to_st,
            "busPosition": bus_pos,
            "dir": v.get("busDirList", [""])[0]
        })
    return vehicles


def fetch_full_route_stations() -> List[Dict[str, Any]]:
    try:
        res = fetch_gbis_route_cached(force_refresh=True)
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

        print(f"[OK] 7790 전 노선 {len(stations)}개 정류장 로드 완료!")
        return stations
    except Exception as e:
        print(f"[ERROR] 정류소 목록 로드 실패: {e}")
        return []


def calc_realtime_traffic_delays(direction: str = "UP") -> List[Dict[str, Any]]:
    """
    GBIS 7790번 66개 구간별 실시간 통과 소요시간(ptts)과 실시간 날씨를 연동하여
    지연 요인(SHAP 기여도)을 실시간으로 산출합니다 (12초 인메모리 캐싱 적용).
    """
    global _CACHE_TRAFFIC_DELAYS, _CACHE_WEATHER
    now = time.time()
    cached = _CACHE_TRAFFIC_DELAYS.get(direction)
    if cached and (now - cached["timestamp"] < TRAFFIC_CACHE_TTL):
        return cached["data"]

    with _LOCK_TRAFFIC_DELAYS:
        now = time.time()
        cached = _CACHE_TRAFFIC_DELAYS.get(direction)
        if cached and (now - cached["timestamp"] < TRAFFIC_CACHE_TTL):
            return cached["data"]

        try:
            res = fetch_gbis_route_cached()
            est = res.get("result", {}).get("estimateTime", [])
        except Exception:
            est = []

        order_ptts = {}
        for item in est:
            try:
                order = int(item.get("staOrder", 0))
                ptts = int(item.get("ptts", 0))
                order_ptts[order] = ptts
            except (ValueError, TypeError):
                pass

        with _LOCK_WEATHER:
            if _CACHE_WEATHER["data"] and (now - _CACHE_WEATHER["timestamp"] < WEATHER_CACHE_TTL):
                weather_shap, weather_desc, weather_pct = _CACHE_WEATHER["data"]
            else:
                weather_shap = "+0.0분"
                weather_desc = "정상 노면 상태"
                weather_pct = 5
                try:
                    rw = HTTP_SESSION.get(
                        "https://api.open-meteo.com/v1/forecast?latitude=37.21&longitude=126.97&current=temperature_2m,precipitation",
                        timeout=3
                    )
                    cw = rw.json().get("current", {})
                    rain_val = float(cw.get("precipitation", 0.0))
                    temp_val = float(cw.get("temperature_2m", 15.0))
                    if rain_val > 0.5:
                        weather_shap = f"+{round(rain_val * 0.8, 1)}분"
                        weather_desc = f"실시간 강수 {rain_val}mm (노면 미끄럼 감속)"
                        weather_pct = 15
                    else:
                        weather_desc = f"기온 {temp_val}℃ (노면 양호, 시야 확보)"
                    _CACHE_WEATHER["data"] = (weather_shap, weather_desc, weather_pct)
                    _CACHE_WEATHER["timestamp"] = time.time()
                except Exception:
                    pass

        if direction == "DOWN":
            sadang_sec = sum(order_ptts.get(o, 60) for o in range(32, 35))
            sadang_delay = round(max(0.5, (sadang_sec - 180) / 60), 1)

            nam_sec = sum(order_ptts.get(o, 70) for o in range(35, 38))
            nam_delay = round(max(0.7, (nam_sec - 210) / 60), 1)

            toll_sec = sum(order_ptts.get(o, 80) for o in range(38, 48))
            toll_delay = round(max(0.8, (toll_sec - 550) / 60), 1)

            city_sec = sum(order_ptts.get(o, 50) for o in range(48, 66))
            city_delay = round(max(0.5, (city_sec - 500) / 60), 1)

            total = sadang_delay + nam_delay + toll_delay + city_delay + 0.3
            result = [
                {"factor": "사당역 9번 출구 승객 승하차 지연", "shapValue": f"+{sadang_delay}분", "percentage": int(sadang_delay / total * 100), "color": "bg-rose-500", "desc": f"사당역 주변 실시간 정차 {sadang_sec}초 집계"},
                {"factor": "과천대로 - 남태령 퇴근길 통행량", "shapValue": f"+{nam_delay}분", "percentage": int(nam_delay / total * 100), "color": "bg-amber-500", "desc": f"남태령 실시간 통과 소요 {nam_sec}초"},
                {"factor": "의왕과천로 (학의JC - 의왕TG) 주행 감속", "shapValue": f"+{toll_delay}분", "percentage": int(toll_delay / total * 100), "color": "bg-blue-500", "desc": f"고속화도로 실시간 주행 {toll_sec}초"},
                {"factor": "봉담IC 진출 및 봉담 시내 신호 대기", "shapValue": f"+{city_delay}분", "percentage": int(city_delay / total * 100), "color": "bg-indigo-500", "desc": f"봉담 시내 실시간 통과 {city_sec}초"},
                {"factor": "실시간 기상 및 도로 노면 상태", "shapValue": weather_shap, "percentage": weather_pct, "color": "bg-emerald-500", "desc": weather_desc}
            ]
        else:
            toll_sec = sum(order_ptts.get(o, 80) for o in range(19, 29))
            toll_delay = round(max(0.8, (toll_sec - 550) / 60), 1)

            sadang_sec = sum(order_ptts.get(o, 80) for o in range(29, 32))
            sadang_delay = round(max(0.6, (sadang_sec - 200) / 60), 1)

            city_sec = sum(order_ptts.get(o, 50) for o in range(1, 19))
            city_delay = round(max(0.6, (city_sec - 600) / 60), 1)

            dep_delay = round(max(0.4, (order_ptts.get(1, 60) - 40) / 60), 1)

            total = toll_delay + sadang_delay + city_delay + dep_delay + 0.3
            result = [
                {"factor": "봉담과천로 (의왕TG - 청계IC) 실시간 통행량", "shapValue": f"+{toll_delay}분", "percentage": int(toll_delay / total * 100), "color": "bg-rose-500", "desc": f"고속화도로 실시간 통과 {toll_sec}초 집계"},
                {"factor": "과천IC - 남태령 고개 서울 진입 병목", "shapValue": f"+{sadang_delay}분", "percentage": int(sadang_delay / total * 100), "color": "bg-amber-500", "desc": f"사당 방면 고가/남태령 실시간 {sadang_sec}초"},
                {"factor": "봉담 시내 구간 승하차 및 신호 연동", "shapValue": f"+{city_delay}분", "percentage": int(city_delay / total * 100), "color": "bg-blue-500", "desc": f"시내 정류장 구간 실시간 {city_sec}초"},
                {"factor": "협성대 기점 출발 정시성 요인", "shapValue": f"+{dep_delay}분", "percentage": int(dep_delay / total * 100), "color": "bg-indigo-500", "desc": f"기점 출차 지표 {order_ptts.get(1, 60)}초"},
                {"factor": "실시간 기상 및 도로 노면 상태", "shapValue": weather_shap, "percentage": weather_pct, "color": "bg-emerald-500", "desc": weather_desc}
            ]

        _CACHE_TRAFFIC_DELAYS[direction] = {"data": result, "timestamp": time.time()}
        return result

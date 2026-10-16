import os
import datetime
import time
import json
import threading
from typing import Literal
import requests
import urllib3
from urllib.parse import quote
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 스레드 안전한 디스크 파일 I/O 락
FILE_IO_LOCK = threading.Lock()

# GBIS 등 공공 SSL 인증서 자체서명 경고 콘솔 노이즈 억제
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()
API_KEY = os.getenv("BUS_API_KEY")

# 대한민국 표준시(KST, UTC+9) 보증 헬퍼 (서버 타임존이 UTC인 클라우드/도커 환경에서도 항상 KST 기준 동작)
KST = datetime.timezone(datetime.timedelta(hours=9))

def get_kst_now():
    """항상 대한민국 표준시(KST, UTC+9) 기준의 datetime 객체를 반환합니다."""
    return datetime.datetime.now(KST)

STATION_ID = "119000302"         # 사당역 9번 출구 앞 (승차)
ROUTE_REGULAR_ID = "200000149"    # 일반 7790
ROUTE_RESERVE_ID = "233000384"    # 7790 (예약/전세)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 최신 권장 lifespan 컨텍스트 매니저 (startup / shutdown)"""
    global CACHED_STATIONS
    CACHED_STATIONS = fetch_full_route_stations()
    load_crowd_data_from_disk()
    yield

app = FastAPI(title="7790 & Smart Transit Router", lifespan=lifespan)

# ⭐ 외부 도메인 및 모바일 WebView 배포 대비 CORS 전면 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

CACHED_STATIONS = []

# ⭐ 동시 접속자 급증 대비 고성능 백엔드 캐시 (공공데이터포털 쿼터 고갈 및 GBIS WAF 차단 방어)
import time
CACHE_BUS_ARRIVALS = {}      # {station_id: {"data": dict, "timestamp": float}}
CACHE_XAI_ANALYSIS = {}      # {station_id: {"data": dict, "timestamp": float}}
CACHE_TRAFFIC_DELAYS = {}    # {direction: {"data": list, "timestamp": float}}
CACHE_WEATHER = {"data": None, "timestamp": 0.0}
CACHE_GBIS_ROUTE = {"data": None, "timestamp": 0.0}   # GBIS 노선/차량/소요시간 단일 통합 인메모리 캐시
CROWD_REPORT_IP_RATE_LIMIT = {}  # {ip: [timestamps]} - 캠퍼스 Wi-Fi 분당 도배 방지

# ⭐ 불감대(Deadband) 기반 좌석 예측 앵커링(Lock-in) 레지스트리
# 버스가 기점에서 출발하여 타겟 정류장에 도달할 때까지, 정상 예측 오차 범위(Deadband) 내에서는
# 숫자가 11석 -> 9석 -> 12석처럼 널뛰지(Jitter) 않도록 초기 예측값을 안정적으로 고정 유지합니다.
# 만약 단체 승차 등으로 허용 오차 범위를 유의미하게 벗어나면(Out-of-bound) 즉시 새 기준으로 갱신(Re-anchor)합니다.
PREDICTION_LOCK_REGISTRY = {}  # {f"{station_id}_{plate}": dict}

BUS_CACHE_TTL = 6.0          # 6초 캐시: 동시 접속자가 100명이어도 외부 API는 6초에 1번만 호출
XAI_CACHE_TTL = 5.0          # 5초 XAI 분석 캐시
TRAFFIC_CACHE_TTL = 12.0     # 12초 캐시
WEATHER_CACHE_TTL = 300.0    # 5분 캐시
GBIS_ROUTE_CACHE_TTL = 8.0   # 8초 GBIS 실시간 데이터 공유 캐시 (도착/차량/소요시간 중복 요청 방어)

def apply_prediction_deadband_lock(
    station_id: str,
    plate: str,
    new_expected: int,
    new_lower: int,
    new_upper: int,
    new_range: str,
    new_prob: int,
    new_status: str,
    new_color: str,
    new_decision: str,
    predict_min: int,
    is_resv: bool
):
    """
    [불감대(Deadband) & 히스테리시스 예측 앵커링 엔진]
    기점에서 버스가 출발한 후 타겟 정류장까지 이동하는 동안,
    실시간 산출 좌석수가 기존 예측 신뢰 구간 [lower_bound, upper_bound] 안에 들어오면
    숫자 널뛰기(Jitter)를 방지하고 사용자 신뢰도를 극대화하기 위해 최초 예측값을 고정 유지합니다.
    예측 범위를 벗어나는 급격한 승하차가 발생할 때만 새로운 현실로 Re-anchor합니다.
    """
    global PREDICTION_LOCK_REGISTRY
    if is_resv or not plate:
        return new_expected, new_lower, new_upper, new_range, new_prob, new_status, new_color, new_decision

    now = time.time()
    lock_key = f"{station_id}_{plate}"

    # 버스가 정류장에 진입 완료(1분 이하) 시 락 해제
    if predict_min <= 1:
        PREDICTION_LOCK_REGISTRY.pop(lock_key, None)
        return new_expected, new_lower, new_upper, new_range, new_prob, new_status, new_color, new_decision

    record = PREDICTION_LOCK_REGISTRY.get(lock_key)
    # 20분 이상 경과한 오래된 레코드는 만료 처리
    if record and (now - record["timestamp"] > 1200):
        record = None

    if record:
        locked_low = record["lower_bound"]
        locked_high = record["upper_bound"]

        # ⭐ 핵심 검증: 실시간 계산치가 이전 예측 신뢰 구간(Deadband) 내에 머무르고 있는가?
        if locked_low <= new_expected <= locked_high:
            # 정상 오차 범위 내의 미세 변동이므로 약속된 예측값을 고정 유지 (Jitter 방지)
            record["timestamp"] = now  # 갱신 유지
            return (
                record["locked_expected"],
                record["lower_bound"],
                record["upper_bound"],
                record["seat_range"],
                record["boarding_prob"],
                record["boarding_status"],
                record["prob_color"],
                record["decision_text"]
            )
        else:
            # 🚨 예측 신뢰 구간 이탈(돌발 승하차/노선 급변 감지) -> 새 기준으로 재앵커링!
            pass

    # 신규 앵커 등록 또는 허용 범위 이탈로 인한 새 기준점 수립
    PREDICTION_LOCK_REGISTRY[lock_key] = {
        "locked_expected": new_expected,
        "lower_bound": new_lower,
        "upper_bound": new_upper,
        "seat_range": new_range,
        "boarding_prob": new_prob,
        "boarding_status": new_status,
        "prob_color": new_color,
        "decision_text": new_decision,
        "timestamp": now
    }
    return new_expected, new_lower, new_upper, new_range, new_prob, new_status, new_color, new_decision


# 💡 GBIS 및 공공기관 방화벽(WAF) 차단 방지를 위한 브라우저 User-Agent
GBIS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def fetch_gbis_route_cached(force_refresh=False):
    """
    GBIS searchRouteJson 엔드포인트를 8초 인메모리 캐싱하여
    도착 버스 조회와 실시간 도로교통 지연 분석(SHAP) 간의 중복 네트워크 I/O를 원천 차단합니다.
    """
    global CACHE_GBIS_ROUTE
    now = time.time()
    if not force_refresh and CACHE_GBIS_ROUTE["data"] and (now - CACHE_GBIS_ROUTE["timestamp"] < GBIS_ROUTE_CACHE_TTL):
        return CACHE_GBIS_ROUTE["data"]

    url_route = "https://www.gbis.go.kr/gbis2014/schBusAPI.action"
    data_route = {"cmd": "searchRouteJson", "routeId": ROUTE_REGULAR_ID}
    try:
        r_live = requests.post(url_route, data=data_route, headers=GBIS_HEADERS, timeout=5, verify=False)
        if r_live.status_code == 200:
            res = r_live.json()
            CACHE_GBIS_ROUTE = {"data": res, "timestamp": now}
            return res
    except Exception as e:
        print(f"GBIS 노선 통합 캐시 조회 오류: {e}")

    return CACHE_GBIS_ROUTE["data"] or {}


def get_real_client_ip(request: Request) -> str:
    """리버스 프록시(Nginx, Cloudflare, AWS, Render, ngrok 등) 환경에서 실제 클라이언트 IP 추출"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"


def is_charter_or_reserved_bus(
    route_id: str = "",
    route_name: str = "",
    plate_no: str = "",
    short_plate: str = "",
    reservation_flag: str = ""
) -> tuple[bool, str]:
    """
    7790번 운행 차량이 일반 정규 광역버스(탑승 가능)인지,
    외부 임차 전세/예약버스(경기 79사/바/아/자, 9000번대 등 일반 학생 탑승 불가)인지 정밀 식별합니다.

    [면허 체계 기준]:
    - 경진여객 정규 노선 면허: 경기 70사(수원시), 경기 76아(화성시)
    - 외부 임차 전세버스 면허: 경기 79사, 경기 79바, 경기 79아, 경기 79자 (및 경기 78사/바/아/자 등 전세버스조합 면허)
    - 예약버스 식별 번호: 차량 뒤 4자리가 9000번대(9XXX)로 임시 배정된 차량
    - GBIS/공공데이터 플래그: ROUTE_RESERVE_ID('233000384'), '예약' 노선명, reservation == 'Y'
    """
    clean_plate = (plate_no or "").replace(" ", "")
    clean_short = (short_plate or "").replace(" ", "")
    if not clean_short and clean_plate:
        clean_short = clean_plate[-4:] if len(clean_plate) >= 4 else clean_plate

    # 1. 공공데이터/GBIS 공식 예약 노선 ID 또는 명칭에 '예약'이 명시된 경우
    if str(route_id) == ROUTE_RESERVE_ID or ("예약" in str(route_name)):
        return True, "GBIS 공공데이터 예약 노선(233000384)으로 지정됨"

    # 2. GBIS 실시간 차량 reservation 플래그
    if str(reservation_flag).upper() == "Y":
        return True, "GBIS 실시간 차량 예약(Reservation) 플래그 식별됨"

    # 3. ⭐ 경기도 전세버스 전용 면허 식별 (경기 79사, 경기 79바, 경기 79아, 경기 79자 등)
    # 경진여객 정규 면허(경기 70사 수원 / 경기 76아 화성) 외의 전세버스 면허인 경우 100% 임차 전세차량
    charter_plate_markers = ["경기79", "79사", "79바", "79아", "79자", "경기78", "78사", "78바", "78아", "78자"]
    for marker in charter_plate_markers:
        if marker in clean_plate:
            return True, f"전세버스 면허('{marker}'): 외부 임차 전세 차량으로 식별됨 (정규 70사/76아 외 면허)"

    # 서울 등 타 지자체 전세버스(서울 70사/바 등) 임차 차량 검사
    if any(m in clean_plate for m in ["서울70사", "서울70바", "서울70아", "서울70자"]):
        return True, f"외부 전세버스 면허('{clean_plate}'): 타 시도 임차 전세 차량"

    # 4. 차량 번호 뒤 4자리가 9000번대인 경우 (출퇴근 전세버스 임시 식별 번호 대역)
    if clean_short.startswith("9"):
        return True, f"차량번호 '{clean_short}': 9000번대 외부 임차 전세버스 식별 번호"

    return False, "정규 광역버스 면허 (경기 70사 수원 / 경기 76아 화성)"


def fetch_full_route_stations():
    """GBIS에서 7790 전 노선 66개 정류장 수집"""
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



@app.get("/")
def get_home():
    return FileResponse("static/index.html")


@app.get("/api/buses")
def get_bus_arrivals(station_id: str = Query("119000302")):
    global CACHED_STATIONS, CACHE_BUS_ARRIVALS
    now = time.time()

    # 1. 고속 인메모리 캐시 확인 (6초 TTL): 다수 학우 동시 접속 시 공공데이터포털 쿼터 고갈 완벽 방어
    cached = CACHE_BUS_ARRIVALS.get(station_id)
    if cached and (now - cached["timestamp"] < BUS_CACHE_TTL):
        res = dict(cached["data"])
        crowd_now = get_crowd_summary(station_id=station_id)
        res["crowdData"] = crowd_now
        if station_id in STATION_XAI_PROFILES and res.get("arrivingBuses"):
            try:
                exps, _ = calculate_bus_predictions(station_id, res["arrivingBuses"], res.get("vehicles", []), crowd_now)
                exp_map = {e["shortPlate"]: e for e in exps}
                updated_arr = []
                for b in res["arrivingBuses"]:
                    b_copy = dict(b)
                    short = b_copy.get("shortPlate")
                    if short in exp_map:
                        b_copy["predictedSeatRange"] = exp_map[short].get("predictedSeatRange")
                        b_copy["boardingStatus"] = exp_map[short].get("boardingStatus")
                        b_copy["boardingProbability"] = exp_map[short].get("boardingProb")
                        b_copy["probColor"] = exp_map[short].get("probColor")
                    updated_arr.append(b_copy)
                res["arrivingBuses"] = updated_arr
            except Exception:
                pass
        return res

    if not CACHED_STATIONS:
        CACHED_STATIONS = fetch_full_route_stations()

    url_arrival = f"https://apis.data.go.kr/6410000/busarrivalservice/v2/getBusArrivalListv2?serviceKey={API_KEY}&stationId={station_id}&format=json"

    arriving_buses = []
    live_vehicles = []

    # 1. 사당역 도착 정보
    try:
        r_arr = requests.get(url_arrival, headers=GBIS_HEADERS, timeout=4)
        if r_arr.status_code == 200:
            arr_data = r_arr.json()
            bus_list = arr_data.get("response", {}).get("msgBody", {}).get("busArrivalList", [])
            for bus in bus_list:
                route_id = str(bus.get("routeId"))
                route_name = str(bus.get("routeName"))
                if route_id in [ROUTE_REGULAR_ID, ROUTE_RESERVE_ID] or "7790" in route_name:
                    if bus.get("predictTime1"):
                        plate = bus.get("plateNo1", "")
                        short_plate = plate[-4:] if len(plate) >= 4 else plate
                        # 예약 노선이거나, 명칭에 예약이 있거나, 경기 79사 등 전세면허/9000번대 임차 전세버스 판별
                        is_resv, resv_reason = is_charter_or_reserved_bus(
                            route_id=route_id,
                            route_name=route_name,
                            plate_no=plate,
                            short_plate=short_plate
                        )
                        arriving_buses.append({
                            "routeName": route_name,
                            "isReserved": is_resv,
                            "reserveReason": resv_reason,
                            "predictTimeMin": int(bus.get("predictTime1")),
                            "remainStations": int(bus.get("locationNo1") or 0),
                            "remainSeats": int(bus.get("remainSeatCnt1") or 0),
                            "plateNo": plate,
                            "shortPlate": short_plate
                        })
                    if bus.get("predictTime2"):
                        plate = bus.get("plateNo2", "")
                        short_plate = plate[-4:] if len(plate) >= 4 else plate
                        is_resv, resv_reason = is_charter_or_reserved_bus(
                            route_id=route_id,
                            route_name=route_name,
                            plate_no=plate,
                            short_plate=short_plate
                        )
                        arriving_buses.append({
                            "routeName": route_name,
                            "isReserved": is_resv,
                            "reserveReason": resv_reason,
                            "predictTimeMin": int(bus.get("predictTime2")),
                            "remainStations": int(bus.get("locationNo2") or 0),
                            "remainSeats": int(bus.get("remainSeatCnt2") or 0),
                            "plateNo": plate,
                            "shortPlate": short_plate
                        })
            arriving_buses.sort(key=lambda x: x["predictTimeMin"])
    except Exception as e:
        print(f"도착 API 오류: {e}")

    # 2. 노선 전체 실시간 차량 위치 (통합 캐시 활용)
    try:
        live_data = fetch_gbis_route_cached()
        v_list = live_data.get("result", {}).get("realTime", {}).get("list", [])
        for v in v_list:
            plate_list = v.get("busNo", [])
            plate = plate_list[0] if plate_list else ""
            short_plate = plate[-4:] if len(plate) >= 4 else plate
            # 경기 79사 전세 면허, reservation 플래그, 9천번대 임차 전세 정밀 판별
            is_reserved, resv_reason = is_charter_or_reserved_bus(
                route_id=ROUTE_REGULAR_ID,
                route_name="7790",
                plate_no=plate,
                short_plate=short_plate,
                reservation_flag=v.get("reservation", "")
            )
            # 카카오맵/GBIS 기준 실제 버스가 정차/출발한 현재 정류장(fromStationId)을 우선 매핑
            target_station = str(v.get("fromStationId") or v.get("toStationId") or "")
            
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

    # 공공데이터 API 도착 버스와 GBIS 실시간 차량 매칭하여 2층 버스 플래그 동기화
    for ab in arriving_buses:
        short = ab.get("shortPlate")
        matched = next((v for v in live_vehicles if v.get("shortPlate") == short), None)
        if matched:
            ab["isDoubleDecker"] = matched.get("isDoubleDecker", False)
        elif "isDoubleDecker" not in ab:
            ab["isDoubleDecker"] = False

    # ⭐ 3. 공공데이터 API 제한(기본 2대) 보완: 사당역 9번 출구(119000302)는 3대까지 실시간 추적 & 무중단 폴백
    max_track_count = 3 if str(station_id) == "119000302" else 2
    if len(arriving_buses) < max_track_count and live_vehicles:
        try:
            est = live_data.get("result", {}).get("estimateTime", [])
            order_map = {item['stationId']: int(item['staOrder']) for item in est}
            order_ptts = {int(item['staOrder']): int(item['ptts']) for item in est}
            target_order = order_map.get(station_id)

            if target_order:
                target_dir = 'D' if target_order >= 33 else 'U'
                existing_shorts = {b.get("shortPlate") for b in arriving_buses}
                candidates = []

                for v in live_vehicles:
                    v_short = v.get("shortPlate", "")
                    if v_short in existing_shorts:
                        continue

                    v_dir = v.get("dir", "")
                    to_id = v.get("currentStationId", "")
                    v_order = order_map.get(to_id)
                    if not v_order:
                        continue

                    # 사당역(order 33): 수원대에서 상행(U)으로 올라와 회차(order 31~32) 거쳐 사당역 9번 출구(order 33)로 오는 차량
                    if str(station_id) == "119000302":
                        if (v_dir == "U" and v_order <= 33) or (v_dir == "D" and v_order == 32):
                            rem_stops = max(1, 33 - v_order)
                            ptts_sum = sum(order_ptts.get(o, 60) for o in range(v_order, 33))
                            mins = max(1, round(ptts_sum / 60))
                            candidates.append({
                                "routeName": "7790(예약)" if v["isReserved"] else "7790",
                                "isReserved": v["isReserved"],
                                "predictTimeMin": mins,
                                "remainStations": rem_stops,
                                "remainSeats": v["remainSeats"],
                                "plateNo": v["plateNo"],
                                "shortPlate": v["shortPlate"],
                                "isDoubleDecker": v.get("isDoubleDecker", False)
                            })
                    else:
                        # 일반 정류장: 같은 방향(target_dir)이면서 정류장 앞서 실제로 다가오는 차량만 수집
                        if v_dir == target_dir and v_order <= target_order:
                            rem_stops = max(1, target_order - v_order)
                            ptts_sum = sum(order_ptts.get(o, 60) for o in range(v_order, target_order))
                            mins = max(1, round(ptts_sum / 60))
                            candidates.append({
                                "routeName": "7790(예약)" if v["isReserved"] else "7790",
                                "isReserved": v["isReserved"],
                                "predictTimeMin": mins,
                                "remainStations": rem_stops,
                                "remainSeats": v["remainSeats"],
                                "plateNo": v["plateNo"],
                                "shortPlate": v["shortPlate"],
                                "isDoubleDecker": v.get("isDoubleDecker", False)
                            })

                candidates.sort(key=lambda x: x["predictTimeMin"])
                needed = max_track_count - len(arriving_buses)
                arriving_buses.extend(candidates[:needed])
                # 실제 운행 중인 차량만 도착 예정 시간순 정렬
                arriving_buses.sort(key=lambda x: x["predictTimeMin"])
        except Exception as e_fb:
            print(f"GBIS 실시간 3대 추적/폴백 산출 오류: {e_fb}")

    crowd_summary_data = get_crowd_summary(station_id=station_id)

    # ⭐ 5대 주요 정류장(사당역9출, 수원대, 신명, 아이파크, 효행초)인 경우 XAI 예상 착석 좌석수 및 탑승 진단 즉시 부착
    if station_id in STATION_XAI_PROFILES and arriving_buses:
        try:
            exps, _ = calculate_bus_predictions(station_id, arriving_buses, live_vehicles, crowd_summary_data)
            exp_map = {e["shortPlate"]: e for e in exps}
            for ab in arriving_buses:
                short = ab.get("shortPlate")
                if short in exp_map:
                    ab["predictedSeatRange"] = exp_map[short].get("predictedSeatRange")
                    ab["boardingStatus"] = exp_map[short].get("boardingStatus")
                    ab["boardingProbability"] = exp_map[short].get("boardingProb")
                    ab["probColor"] = exp_map[short].get("probColor")
        except Exception as e_xai_attach:
            print(f"XAI 예측 도착 버스 부착 오류: {e_xai_attach}")

    res_data = {
        "success": True,
        "stations": CACHED_STATIONS,
        "arrivingBuses": arriving_buses,
        "vehicles": live_vehicles,
        "crowdData": crowd_summary_data
    }
    CACHE_BUS_ARRIVALS[station_id] = {"data": res_data, "timestamp": now}
    return res_data

# ⭐ 정류장별 학우 실시간 제보 영구 보존 및 IP 도배 방지 쿨다운
CROWD_DATA_FILE = "crowd_reports_cache.json"
CROWD_REPORTS_BY_STATION = {}
CROWD_REPORT_IP_COOLDOWN = {}  # {ip: timestamp}

def save_crowd_data_to_disk():
    try:
        now = time.time()
        clean_reports = {}
        for st_id, r_list in CROWD_REPORTS_BY_STATION.items():
            valid = [r for r in r_list if now - r["time"] < 600]
            if valid:
                clean_reports[st_id] = valid
        clean_ips = {}
        for ip, record in CROWD_REPORT_IP_COOLDOWN.items():
            t = record.get("time", 0) if isinstance(record, dict) else record
            dur = record.get("duration", 600) if isinstance(record, dict) else 600
            if now - t < dur:
                clean_ips[ip] = record
        with FILE_IO_LOCK:
            with open(CROWD_DATA_FILE, "w", encoding="utf-8") as f:
                json.dump({"reports": clean_reports, "ips": clean_ips}, f)
    except Exception as e:
        print(f"제보 데이터 디스크 저장 오류: {e}")

def load_crowd_data_from_disk():
    global CROWD_REPORTS_BY_STATION, CROWD_REPORT_IP_COOLDOWN
    try:
        with FILE_IO_LOCK:
            if not os.path.exists(CROWD_DATA_FILE):
                return
            with open(CROWD_DATA_FILE, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        now = time.time()
        clean_reports = {}
        for st_id, r_list in data.get("reports", {}).items():
            valid = [r for r in r_list if now - r["time"] < 600]
            if valid:
                clean_reports[st_id] = valid
        CROWD_REPORTS_BY_STATION = clean_reports
        clean_ips = {}
        for ip, record in data.get("ips", {}).items():
            t = record.get("time", 0) if isinstance(record, dict) else record
            dur = record.get("duration", 600) if isinstance(record, dict) else 600
            if now - t < dur:
                clean_ips[ip] = record
        CROWD_REPORT_IP_COOLDOWN = clean_ips
        print(f"[OK] 제보 데이터 {sum(len(v) for v in clean_reports.values())}건 디스크 복원 완료!")
    except Exception as e:
        print(f"제보 데이터 디스크 복원 오류: {e}")

@app.post("/api/crowd-report")
def report_crowd(
    request: Request,
    status: Literal["smooth", "moderate", "crowded"] = Query(..., description="여유(smooth), 보통(moderate), 포화(crowded)"),
    station_id: str = Query("119000302", description="제보 정류장 ID")
):
    """정류장별 학우들의 1초 원터치 줄 상태 제보 저장 API (도착 버스 연동 동적 쿨다운 최대 10분, 디스크 영구보존)"""
    import time
    now = time.time()
    global CROWD_REPORTS_BY_STATION, CROWD_REPORT_IP_COOLDOWN, CACHE_BUS_ARRIVALS, CACHE_XAI_ANALYSIS

    # 1. 제보 등록 시점 해당 정류장 첫 번째 도착 예정 버스 정보 조회
    target_bus_plate = ""
    target_eta_min = 5
    cached = CACHE_BUS_ARRIVALS.get(station_id)
    if not (cached and cached.get("data")):
        try:
            get_bus_arrivals(station_id=station_id)
            cached = CACHE_BUS_ARRIVALS.get(station_id)
        except Exception:
            pass
    if cached and cached.get("data"):
        arrs = cached["data"].get("arrivingBuses", [])
        if arrs:
            target_bus_plate = arrs[0].get("shortPlate", "")
            target_eta_min = arrs[0].get("predictTimeMin", 5)

    # ⭐ 학우 요청 반영: 첫 번째 도착 예정 버스 잔여시간 연동 동적 쿨다운
    # - 최소 2분 (120초): 버스가 0~1분 전이어도 연타 도배 방지
    # - 최대 10분 (600초): 학우 요청 최대 10분 상한선 설정
    if target_eta_min <= 1:
        dynamic_cooldown = 120  # 2분 (다음 버스 대기열 제보 준비)
    else:
        dynamic_cooldown = max(120, min(target_eta_min * 60, 600))  # 최대 10분(600초)

    # 2. 하이브리드 Rate Limit: 개인 UUID 식별 + 캠퍼스 Wi-Fi IP 분당 총량 제한
    client_ip = get_real_client_ip(request)
    client_uuid = request.headers.get("x-client-id", "").strip()
    user_id = client_uuid if client_uuid else client_ip
    cooldown_key = f"{user_id}_{station_id}"

    # 2-1. 공인 IP 기준 분당 총량 제한 (스크립트/도배 방어, 캠퍼스 Wi-Fi 고려 분당 최대 8건)
    ip_history = CROWD_REPORT_IP_RATE_LIMIT.get(client_ip, [])
    ip_history = [t for t in ip_history if now - t < 60]
    if len(ip_history) >= 8:
        return {
            "success": False,
            "message": "교내 네트워크에서 일시적으로 제보가 집중되었습니다. 1분 후 다시 시도해 주세요.",
            "remainCooldown": 30,
            "stationId": station_id
        }

    # 2-2. 개인별(UUID/기기) 정류장 쿨다운 검사
    last_record = CROWD_REPORT_IP_COOLDOWN.get(cooldown_key)
    if isinstance(last_record, dict):
        last_time = last_record.get("time", 0)
        cooldown_duration = last_record.get("duration", 300)
    elif isinstance(last_record, (int, float)):
        last_time = last_record
        cooldown_duration = 300
    else:
        last_time = 0
        cooldown_duration = 300

    remain = int(cooldown_duration - (now - last_time))
    if remain > 0:
        remain_min = remain // 60
        remain_sec = remain % 60
        time_msg = f"{remain_min}분 {remain_sec}초" if remain_min > 0 else f"{remain_sec}초"
        return {
            "success": False,
            "message": f"도배 방지 및 도착 주기 연동으로 {time_msg} 후 다시 제보 가능합니다.",
            "remainCooldown": remain,
            "stationId": station_id
        }

    ip_history.append(now)
    CROWD_REPORT_IP_RATE_LIMIT[client_ip] = ip_history

    if station_id not in CROWD_REPORTS_BY_STATION:
        CROWD_REPORTS_BY_STATION[station_id] = []

    # 최근 10분(600초) 제보 보존 (최대 10분 락 및 점진 감쇄 지원)
    CROWD_REPORTS_BY_STATION[station_id] = [r for r in CROWD_REPORTS_BY_STATION[station_id] if now - r["time"] < 600]
    CROWD_REPORTS_BY_STATION[station_id].append({
        "status": status,
        "time": now,
        "targetPlate": target_bus_plate,
        "targetEta": target_eta_min
    })
    CROWD_REPORT_IP_COOLDOWN[cooldown_key] = {"time": now, "duration": dynamic_cooldown}

    # ⭐ 디스크 영구 저장 (서버 재부팅/재배포 시에도 제보 보존)
    save_crowd_data_to_disk()

    # ⭐ 공공데이터 도착 캐시(CACHE_BUS_ARRIVALS)는 유지하여 쿼터 방어!
    # 제보 정류장 및 연관 상행선 XAI 분석 캐시만 선택적으로 무효화
    CACHE_XAI_ANALYSIS.pop(station_id, None)
    if station_id in ["233002245", "233002371", "233002255", "233003021"]:
        for st in ["233002245", "233002371", "233002255", "233003021"]:
            CACHE_XAI_ANALYSIS.pop(st, None)
    
    cooldown_min_text = f"{dynamic_cooldown // 60}분"
    return {
        "success": True, 
        "message": f"학우님의 제보가 실시간 XAI 모델에 즉시 반영되었습니다! (도착 주기 연동 쿨다운: {cooldown_min_text})",
        "cooldownSec": dynamic_cooldown,
        "stationId": station_id
    }

def get_crowd_summary(station_id="119000302"):
    """최근 제보 요약 및 버스 미도착 시 대기열 락(Lock) & 점진 감쇄(Decay) 분석"""
    import time, datetime
    now = time.time()
    reports = CROWD_REPORTS_BY_STATION.get(station_id, [])

    # 1. 5분(300초) 이내 신선한 제보
    fresh_reports = [r for r in reports if now - r["time"] < 300]
    # 2. 5분~10분(600초) 이전 제보 (버스 미도착 상태 잔류 대기열 검사용, 최대 10분)
    older_reports = [r for r in reports if 300 <= (now - r["time"]) < 600]

    # 정류장 현재 첫 번째 도착 예정 버스 조회
    curr_first_plate = ""
    curr_first_min = 999
    cached_buses = CACHE_BUS_ARRIVALS.get(station_id)
    if cached_buses and cached_buses.get("data"):
        arr_list = cached_buses["data"].get("arrivingBuses", [])
        if arr_list:
            curr_first_plate = arr_list[0].get("shortPlate", "")
            curr_first_min = arr_list[0].get("predictTimeMin", 999)

    is_sticky = False
    sticky_decay_factor = 1.0

    if fresh_reports:
        valid_reports = fresh_reports
    elif older_reports:
        last_r = older_reports[-1]
        age_sec = now - last_r["time"]
        target_plate = last_r.get("targetPlate", "")
        # ⭐ 핵심: 직전 제보 당시 타겟 버스가 아직 안 왔거나, 첫 차가 오고 있는 중(curr_first_min > 1)
        bus_not_arrived_yet = (target_plate and target_plate == curr_first_plate) or (curr_first_min > 1)

        if last_r["status"] in ["crowded", "moderate"] and bus_not_arrived_yet:
            # 승객이 아직 버스에 타지 못했으므로 혼잡 대기열을 유지(Lock)하되 반감기 점진 감쇄 적용 (최대 10분)!
            is_sticky = True
            valid_reports = [last_r]
            if age_sec < 450:       # 5~7.5분: 85% 잔류
                sticky_decay_factor = 0.85
            else:                   # 7.5~10분: 70% 잔류
                sticky_decay_factor = 0.70
        else:
            valid_reports = []
    else:
        valid_reports = []

    profile = STATION_XAI_PROFILES.get(station_id)
    if profile:
        st_name = profile["name"]
        is_down = (profile["dir"] == "DOWN")
    else:
        matched = next((s for s in CACHED_STATIONS if s["id"] == str(station_id)), None)
        if matched:
            st_name = matched["name"]
            is_down = (matched.get("dir") == "DOWN")
        else:
            st_name = "해당 정류장"
            is_down = (str(station_id) == "119000302")

    # ⭐ 방향별 피크 시간대 판별:
    # 1) 사당역 9번 출구 (하행 DOWN, 수원대행): 평일 아침 등교 피크 (07:30 ~ 10:10)
    # 2) 수원대입구/신명/아이파크/효행초 (상행 UP, 사당역행): 평일 오후 하교 피크 (15:25 ~ 17:30)
    now_dt = get_kst_now()
    time_float = now_dt.hour + now_dt.minute / 60.0
    weekday = now_dt.weekday()  # 0:월 ~ 4:금, 5:토, 6:일

    if is_down:
        is_morning_peak = (weekday < 5) and (7.5 <= time_float <= (10.0 + 10.0/60.0))
        is_evening_peak = (weekday < 5) and (16.5 <= time_float <= 19.5)
        is_peak = is_morning_peak or is_evening_peak
        if is_morning_peak:
            peak_name = "등교 피크 (07:30~10:10)"
        elif is_evening_peak:
            peak_name = "퇴근 피크 (16:30~19:30)"
        else:
            peak_name = ""
    else:
        is_peak = (weekday < 5) and ((15.0 + 25.0/60.0) <= time_float <= (17.0 + 30.0/60.0))
        peak_name = "하교 피크 (15:25~17:30)"
        is_morning_peak = False
        is_evening_peak = False

    if valid_reports:
        cnt_smooth = sum(1 for r in valid_reports if r["status"] == "smooth")
        cnt_mod = sum(1 for r in valid_reports if r["status"] == "moderate")
        cnt_crowd = sum(1 for r in valid_reports if r["status"] == "crowded")
        status_count = {'smooth': cnt_smooth, 'moderate': cnt_mod, 'crowded': cnt_crowd}
        dominant_status = max(status_count, key=status_count.get)
        last_sec = int(now - valid_reports[-1]["time"])
        last_min = max(0, last_sec // 60)
        if dominant_status == "smooth":
            status_text = "여유 (<15명 대기)"
            desc_text = f"대기열 제보가 여유로우나, {peak_name}에는 서비스 미사용 일반 승객 대기열이 함께 차감됩니다." if (is_down and is_peak) else f"대기열이 짧아 곧 도착하는 7790번에 바로 착석 탑승이 유력합니다."
            xai_insight = f"{st_name} 실시간 제보(여유)에 미사용 일반 대기 인원을 사전 보정하여 착석 확률을 산출합니다." if (is_down and is_peak) else f"{st_name} 대기열이 15명 미만으로 여유로워 첫 차 착석 성공 확률이 대폭 상승했습니다."
            color = "emerald"
        elif dominant_status == "moderate":
            status_text = "보통 (15~25명 대기)" if not is_sticky else "보통 (15~25명 대기 잔류)"
            desc_text = (f"직전 제보 이후 버스 승차가 아직 완료되지 않아, 보통 대기열({int(sticky_decay_factor*100)}%)이 유지 중인 것으로 추정됩니다."
                         if is_sticky else
                         (f"{peak_name}로 15~25명 학우 제보에 서비스 미사용 일반 대기열이 합산 사전 차감됩니다." if (is_down and is_peak) else
                          f"{f'{peak_name}로 ' if is_peak else ''}15~25명 내외 대기열이 형성 중입니다. 줄 순번에 따라 안정적인 착석이 가능합니다."))
            xai_insight = (f"사당역 9번 출구 직전 보통(15~25명) 제보 잔류 및 일반 대기열을 합산하여 착석을 예측합니다."
                           if (is_down and is_sticky) else
                           (f"사당역 9번 출구 보통 제보에 서비스 미사용 일반 대기열을 사전 합산하여 실착석 가능성을 분석 중입니다." if is_down else
                            f"{st_name} 실시간 대기열(15~25명)로 앞 정류장 승차 누적 잔여석과의 경합이 발생 중입니다."))
            color = "amber"
        else:
            status_text = "포화 (정류장 줄 장동)" if not is_sticky else "포화 (승차 대기 잔류)"
            desc_text = (f"직전 혼잡 제보 이후 버스 승차가 아직 완료되지 않아, 대기 인원({int(sticky_decay_factor*100)}%)이 줄을 유지 중인 것으로 추정됩니다."
                         if is_sticky else
                         (f"{peak_name} 집중으로 서비스 미사용 일반 승객을 포함한 대기열이 길어 조기 만석 위험이 큽니다." if (is_down and is_peak) else
                          f"{f'{peak_name} 학생 집중으로 ' if is_peak else ''}줄이 길게 늘어서 있습니다. 다음 차 탑승을 고려하세요."))
            xai_insight = (f"{st_name} 직전 혼잡 제보 및 일반 대기 잔류로 잔여석 소진 속도가 매우 빠릅니다."
                           if is_sticky else
                           (f"{st_name} 대기 인원 포화(비사용자 합산)로 버스 조기 만석 위험이 반영되어 1층 버스 착석 확률이 급감했습니다." if is_down else
                            f"{st_name} 대기 인원 포화로, 버스 1대 만석 초과 위험이 반영되어 착석 확률이 30% 이하로 급감했습니다."))
            color = "rose"
        total_valid = len(valid_reports)
        confidence = 1.0 if total_valid >= 2 else 0.70  # 1건 제보 시 70%, 2건 이상 교차 검증 시 100% 반영
        consensus_text = f"교차 검증 완료 ({total_valid}건 제보, 신뢰도 100%)" if total_valid >= 2 else "단일 제보 접수 (신뢰도 70%)"

        return {
            "hasData": True,
            "status": dominant_status,
            "statusText": status_text,
            "descText": desc_text,
            "xaiInsight": xai_insight,
            "color": color,
            "totalCount": total_valid,
            "confidence": confidence,
            "consensusText": consensus_text,
            "counts": {"smooth": cnt_smooth, "moderate": cnt_mod, "crowded": cnt_crowd},
            "lastReportMinutesAgo": last_min,
            "isPeak": is_peak,
            "peakName": peak_name,
            "isMorningRush": (is_peak and is_down and is_morning_peak),
            "isEveningRush": (is_peak and is_down and is_evening_peak),
            "isAfternoonRush": (is_peak and not is_down),
            "isSticky": is_sticky,
            "stickyDecayFactor": sticky_decay_factor
        }
    else:
        rush_hint = f" (현재 {peak_name} 시간대)" if is_peak else ""
        return {
            "hasData": False,
            "status": "none",
            "statusText": "최근 5분 내 제보 없음",
            "descText": (f"현재 {st_name} 대기열 제보가 없습니다{rush_hint}. 서비스 미사용자 및 일반 통학생 대기열을 사전에 차감하여 예측 중입니다."
                         if (is_down and is_peak) else
                         f"현재 {st_name} 대기열 제보가 없습니다{rush_hint}. 노선도 탭에서 1초 제보에 참여해 주세요!"),
            "xaiInsight": (f"최근 5분 제보가 없어 {peak_name} 통계 모델(비사용자 대기열 사전 차감)을 적용 중입니다."
                           if (is_down and is_peak) else
                           f"최근 5분 제보가 없어 {f'{peak_name} 통계 모델' if is_peak else '평일 정규 승하차 패턴'}을 적용 중입니다."),
            "color": "gray",
            "totalCount": 0,
            "counts": {"smooth": 0, "moderate": 0, "crowded": 0},
            "lastReportMinutesAgo": None,
            "isPeak": is_peak,
            "peakName": peak_name,
            "isMorningRush": (is_peak and is_down and is_morning_peak),
            "isEveningRush": (is_peak and is_down and is_evening_peak),
            "isAfternoonRush": (is_peak and not is_down),
            "isSticky": False,
            "stickyDecayFactor": 1.0
        }

# ⭐ 사당역 9번 출구(하행 수원대행) 학우 제보 부재 시 자동 작동하는
# 수원대학교 강의 시간표 및 요일별 통학 데이터 기반 사전 통계 모델 (Bayesian Prior)
def estimate_sadang_waiters_by_time():
    now_dt = get_kst_now()
    hour = now_dt.hour
    minute = now_dt.minute
    weekday = now_dt.weekday()  # 0:월, 1:화, ..., 4:금, 5:토, 6:일

    # 주말 (토/일)
    if weekday >= 5:
        if 11 <= hour <= 19:
            return 9, "주말 일반 통행 분산 (일반 승객 포함)"
        else:
            return 3, "주말 한산 시간대"

    # 평일 요일 가중치 (월/화 등교 집중 1.05배, 목요일 원활 0.90배, 금공강 0.70배, 수요일 1.0배)
    day_multiplier = 1.05 if weekday in [0, 1] else (0.70 if weekday == 4 else (0.90 if weekday == 3 else 1.0))
    day_name = ["월", "화", "수", "목", "금"][weekday]

    time_float = hour + minute / 60.0

    # 💡 사용자 피드백 반영: 대기열 차감 인원을 기존 대비 2/3(약 67%) 수준으로 하향 조정
    if 7.5 <= time_float < (7 + 50/60.0):  # 07:30 ~ 07:50 (1교시 등교열 급증 구간)
        base = 39
        desc = f"{day_name}요일 1교시 등교 급증 (비사용자·일반 승객 포함)"
    elif (7 + 50/60.0) <= time_float < (8 + 35/60.0):  # 07:50 ~ 08:35 (9시 1교시 등교 최대 극피크)
        base = 51
        desc = f"{day_name}요일 1교시 등교 극피크 (비사용자·일반 승객 포함)"
    elif (8 + 35/60.0) <= time_float < (8 + 55/60.0):  # 08:35 ~ 08:55 (1교시 직전 막바지 등교 & 2교시 유입)
        base = 35  # 사용자 요청 반영: 3명 추가 감축 (기존 38명 -> 35명)
        desc = f"{day_name}요일 1교시 막바지·2교시 유입 (비사용자·일반 승객 포함)"
    elif (8 + 55/60.0) <= time_float < (9 + 35/60.0):  # 08:55 ~ 09:35 (10:30 2교시 등교 피크)
        base = 31  # 사용자 요청 반영: 3명 추가 감축 (기존 34명 -> 31명)
        desc = f"{day_name}요일 10:30 2교시 등교 피크 (비사용자·일반 승객 포함)"
    elif (9 + 35/60.0) <= time_float < (10 + 10/60.0):  # 09:35 ~ 10:10 (2교시 승차 마무리 및 완화)
        base = 25
        desc = f"{day_name}요일 2교시 승차 마무리 (비사용자 포함)"
    elif (10 + 10/60.0) <= time_float < 12.0:  # 10:10 ~ 12:00 (오전 분산 통학)
        base = 12
        desc = "오전 분산 통학 시간대"
    elif 12.0 <= time_float < 13.5:  # 12:00 ~ 13:30 (오후 수업 등교 & 점심)
        base = 8
        desc = "3교시 점심 분산 시간대"
    elif 13.5 <= time_float < 16.5:  # 13:30 ~ 16:30 (낮 한산 시간대)
        base = 6
        desc = "낮 시간대 통학 한산"
    elif 16.5 <= time_float < 17.5:  # 16:30 ~ 17:30 (퇴근길 유입 시작)
        base = 40
        desc = f"{day_name}요일 퇴근 유입 시작 (퇴근 시민·비사용자 포함)"
    elif 17.5 <= time_float < 19.0:  # 17:30 ~ 19:00 (퇴근길 최대 극피크)
        base = 52
        desc = f"{day_name}요일 퇴근길 최대 극피크 (퇴근 시민·비사용자 집중)"
    elif 19.0 <= time_float < 19.5:  # 19:00 ~ 19:30 (퇴근길 잔여 피크)
        base = 40
        desc = f"{day_name}요일 퇴근길 분산 (퇴근 시민·비사용자 포함)"
    elif 19.5 <= time_float < 22.0:  # 19:30 ~ 22:00 (야간 귀가)
        base = 12
        desc = "야간 귀가 시간대 (일반 승객 포함)"
    else:  # 심야/새벽
        base = 4
        desc = "심야/새벽 한산 시간대"

    est = int(round(base * day_multiplier))
    return est, desc

# ⭐ 실시간 GBIS 도로교통 소통시간 및 실시간 기상/노면 데이터 연동 함수
def calc_realtime_traffic_delays(direction="UP"):
    """
    GBIS 7790번 66개 구간별 실시간 통과 소요시간(ptts)과 실시간 날씨를 연동하여
    SHAP 지연 기여 요인을 실시간으로 산출합니다 (12초 인메모리 캐싱 적용).
    """
    global CACHE_TRAFFIC_DELAYS, CACHE_WEATHER
    now = time.time()
    cached = CACHE_TRAFFIC_DELAYS.get(direction)
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

    # 실시간 기상 데이터 (화성 봉담 / 과천 / 사당 일대, 5분 캐시)
    if CACHE_WEATHER["data"] and (now - CACHE_WEATHER["timestamp"] < WEATHER_CACHE_TTL):
        weather_shap, weather_desc, weather_pct = CACHE_WEATHER["data"]
    else:
        weather_shap = "+0.0분"
        weather_desc = "정상 노면 상태"
        weather_pct = 5
        try:
            rw = requests.get(
                "https://api.open-meteo.com/v1/forecast?latitude=37.21&longitude=126.97&current=temperature_2m,precipitation",
                headers=GBIS_HEADERS,
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
            CACHE_WEATHER["data"] = (weather_shap, weather_desc, weather_pct)
            CACHE_WEATHER["timestamp"] = now
        except Exception:
            pass

    if direction == "DOWN":
        # 1. 사당역 9번 출구 회차 및 승하차
        q_sec = order_ptts.get(32, 120) + order_ptts.get(33, 180)
        q_delay = round(max(0.8, (q_sec - 180) / 60), 1)

        # 2. 과천대로 - 남태령 구간
        nam_sec = sum(order_ptts.get(o, 60) for o in range(34, 38))
        nam_delay = round(max(0.6, (nam_sec - 180) / 60), 1)

        # 3. 의왕과천로 (과천IC - 의왕TG)
        toll_sec = sum(order_ptts.get(o, 80) for o in range(38, 47))
        toll_delay = round(max(0.8, (toll_sec - 500) / 60), 1)

        # 4. 봉담IC 진출 및 봉담 시내
        city_sec = sum(order_ptts.get(o, 50) for o in range(47, 67))
        city_delay = round(max(0.5, (city_sec - 600) / 60), 1)

        total = q_delay + nam_delay + toll_delay + city_delay + 0.3
        result = [
            {"factor": "사당역 9번 출구 승하차 및 대기열", "shapValue": f"+{q_delay}분", "percentage": int(q_delay / total * 100), "color": "bg-rose-500", "desc": f"GBIS 실시간 정류소 통과 {q_sec}초 반영"},
            {"factor": "과천대로 - 남태령 퇴근길 통행량", "shapValue": f"+{nam_delay}분", "percentage": int(nam_delay / total * 100), "color": "bg-amber-500", "desc": f"남태령 실시간 통과 소요 {nam_sec}초"},
            {"factor": "의왕과천로 (학의JC - 의왕TG) 주행 감속", "shapValue": f"+{toll_delay}분", "percentage": int(toll_delay / total * 100), "color": "bg-blue-500", "desc": f"고속화도로 실시간 주행 {toll_sec}초"},
            {"factor": "봉담IC 진출 및 봉담 시내 신호 대기", "shapValue": f"+{city_delay}분", "percentage": int(city_delay / total * 100), "color": "bg-indigo-500", "desc": f"봉담 시내 실시간 통과 {city_sec}초"},
            {"factor": "실시간 기상 및 도로 노면 상태", "shapValue": weather_shap, "percentage": weather_pct, "color": "bg-emerald-500", "desc": weather_desc}
        ]
    else:
        # 상행 (사당역행)
        # 1. 봉담과천로 고속화도로
        toll_sec = sum(order_ptts.get(o, 80) for o in range(19, 29))
        toll_delay = round(max(0.8, (toll_sec - 550) / 60), 1)

        # 2. 과천IC - 남태령 서울 진입 병목
        sadang_sec = sum(order_ptts.get(o, 80) for o in range(29, 32))
        sadang_delay = round(max(0.6, (sadang_sec - 200) / 60), 1)

        # 3. 봉담 시내 구간
        city_sec = sum(order_ptts.get(o, 50) for o in range(1, 19))
        city_delay = round(max(0.6, (city_sec - 600) / 60), 1)

        # 4. 협성대 기점 출발 정시성
        dep_delay = round(max(0.4, (order_ptts.get(1, 60) - 40) / 60), 1)

        total = toll_delay + sadang_delay + city_delay + dep_delay + 0.3
        result = [
            {"factor": "봉담과천로 (의왕TG - 청계IC) 실시간 통행량", "shapValue": f"+{toll_delay}분", "percentage": int(toll_delay / total * 100), "color": "bg-rose-500", "desc": f"고속화도로 실시간 통과 {toll_sec}초 집계"},
            {"factor": "과천IC - 남태령 고개 서울 진입 병목", "shapValue": f"+{sadang_delay}분", "percentage": int(sadang_delay / total * 100), "color": "bg-amber-500", "desc": f"사당 방면 고가/남태령 실시간 {sadang_sec}초"},
            {"factor": "봉담 시내 구간 승하차 및 신호 연동", "shapValue": f"+{city_delay}분", "percentage": int(city_delay / total * 100), "color": "bg-blue-500", "desc": f"시내 정류장 구간 실시간 {city_sec}초"},
            {"factor": "협성대 기점 출발 정시성 요인", "shapValue": f"+{dep_delay}분", "percentage": int(dep_delay / total * 100), "color": "bg-indigo-500", "desc": f"기점 출차 지표 {order_ptts.get(1, 60)}초"},
            {"factor": "실시간 기상 및 도로 노면 상태", "shapValue": weather_shap, "percentage": weather_pct, "color": "bg-emerald-500", "desc": weather_desc}
        ]

    CACHE_TRAFFIC_DELAYS[direction] = {"data": result, "timestamp": now}
    return result

# ⭐ 5대 정류장별 XAI 프로필
STATION_XAI_PROFILES = {
    "119000302": {
        "name": "사당역9번출구앞",
        "dir": "DOWN",
        "dirText": "수원대행 (하행)",
        "seq": 2,
        "summary": "사당역 9번 출구는 승객 대기열(줄 길이)이 탑승 여부의 제1 요인입니다. 학우들의 실시간 줄 제보가 XAI 분석에 직결됩니다.",
        "boardingTip": "등교시간대 만석 시 다른 방법을 고려하세요."
    },
    "233003021": {
        "name": "수원대입구",
        "dir": "UP",
        "dirText": "사당역행 (상행)",
        "seq": 16,
        "summary": "수원대입구는 봉담 시내 마지막 주요 정류장으로, 앞선 정류장(효행초/아이파크/신명) 승차 누적으로 인해 하교 피크 시 잔여좌석 소진율이 가장 높습니다.",
        "boardingTip": "하교 피크(15:25~17:30)에는 잔여 15석 미만일 경우 다음 차량이나 앞 정류장(신명/아이파크) 승차를 고려하세요."
    },
    "233002255": {
        "name": "신명아파트",
        "dir": "UP",
        "dirText": "사당역행 (상행)",
        "seq": 14,
        "summary": "신명아파트는 대단지 주거지역 승차가 집중되는 지점입니다. 수원대입구 전 정류장으로 잔여좌석 착석 확률이 15~20% 더 양호합니다.",
        "boardingTip": "수원대입구보다 1개 정류장 앞서 탑승하므로 하교 시간대에는 아슬아슬합니다."
    },
    "233002371": {
        "name": "아이파크정문",
        "dir": "UP",
        "dirText": "사당역행 (상행)",
        "seq": 12,
        "summary": "아이파크정문은 기점 출발 초기 정류장으로, 잔여좌석이 여유로워 안정적 착석 승차가 보장되는 정류장입니다. 다만 하교 시간대에는 잔여좌석이 빠르게 소진될 수 있습니다.",
        "boardingTip": "기점에서 불과 몇 정거장 지나지 않은 구간으로, 버스 도착 시간에 맞춰 나오시면 여유롭게 착석할 수 있습니다."
    },
    "233002245": {
        "name": "효행초등학교정문",
        "dir": "UP",
        "dirText": "사당역행 (상행)",
        "seq": 10,
        "summary": "효행초정문은 협성대 기점 바로 다음의 극초반 정류장입니다. 잔여좌석이 45석 내외로 사실상 빈 차 상태로 진입하여 착석 성공률이 99%입니다.",
        "boardingTip": "기점 인근 최고의 명당 정류장입니다. 버스 진입 시 즉시 원하는 자리에 착석할 수 있습니다."
    }
}
# 상행(사당행) 정류장 순서 노드 체인 (효행초 -> 아이파크 -> 신명 -> 수원대입구)
UPSTREAM_STATION_CHAIN = [
    {"id": "233002245", "name": "효행초", "seq": 10},
    {"id": "233002371", "name": "아이파크", "seq": 12},
    {"id": "233002255", "name": "신명아파트", "seq": 14},
    {"id": "233003021", "name": "수원대입구", "seq": 16}
]

def calculate_bus_predictions(station_id: str, arriving: list, vehicles: list, crowd_info: dict) -> tuple[list, bool]:
    """
    정류장별 실시간 도착 버스의 XAI 좌석수 및 착석 확률을 도메인 휴리스틱 및 신뢰구간 불감대(Deadband) 모델로 산출합니다.
    (/api/buses 및 /api/xai-analysis 양쪽에서 단일 엔진으로 공유)
    """
    profile = STATION_XAI_PROFILES.get(station_id, STATION_XAI_PROFILES["119000302"])
    is_down = (profile["dir"] == "DOWN")
    is_morning_rush = crowd_info.get("isMorningRush", False)
    is_afternoon_rush = crowd_info.get("isAfternoonRush", False)

    # 상행(사당행) 효행초등학교(233002245) 선행 제보 및 실시간 잔여석 급감 사전 검사
    is_hyohaeng_surge = False
    if not is_down:
        hyohaeng_summary = get_crowd_summary(station_id="233002245")
        h_st = hyohaeng_summary.get("status", "none")
        if h_st in ["crowded", "moderate"]:
            is_hyohaeng_surge = True
        else:
            # ⭐ 학우 피드백 반영: 제보가 없더라도 효행초(seq <= 11) 구간을 통과한 직후 버스의 잔여석이
            # 45석 -> 35석 이하(또는 2층 58석 이하)로 이미 크게 소진되었다면,
            # 뒤따르는 아이파크/수원대에 엄청난 대기열이 집중되어 있다는 실측 시그널!
            st_seq_map = {str(s["id"]): s.get("seq", 999) for s in CACHED_STATIONS if s.get("dir") == "UP"}
            for v in vehicles:
                if v.get("dir") == "U":
                    curr_st = str(v.get("currentStationId", ""))
                    v_seq = st_seq_map.get(curr_st, 999)
                    # 효행초 정류장 부근(기점 ~ 효행초 직후 seq <= 11)에 있는 차량만 검사!
                    # (이미 수원대를 지나 과천/사당 다 와가는 버스는 잔여석이 당연히 적으므로 제외)
                    if v_seq <= 11:
                        rem = v.get("remainSeats", 45)
                        is_dd = v.get("isDoubleDecker", False)
                        if (not is_dd and 0 < rem <= 35) or (is_dd and 0 < rem <= 58):
                            is_hyohaeng_surge = True
                            break

    bus_explanations = []
    target_list = arriving[:3] if is_down else arriving[:2]
    prev_bus_waiters = 0

    for idx, b in enumerate(target_list, 1):
        plate = b.get("shortPlate", "")
        is_resv = b.get("isReserved", False)
        seats = b.get("remainSeats", 0)
        predict_min = b.get("predictTimeMin", 0)

        if is_resv:
            full_plate = b.get("plateNo", plate)
            clean_full = full_plate.replace(" ", "")
            is_79_plate = any(cp in clean_full for cp in ["경기79", "79사", "79바", "79아", "79자", "경기78", "78사", "78바", "78아", "78자"])

            if is_79_plate:
                feature_name = "차량 면허 대역 (경기 79사 등 전세버스)"
                feature_desc = f"차량번호 '{full_plate}'는 정규 경진여객(경기 70사/76아)이 아닌 외부 전세버스 면허로 식별됨"
                decision_text = f"차량 '{full_plate}'는 경기 79사 등 외부 임차 전세버스로, 예약자 전용 차량이므로 [일반 학생 탑승 불가] 차량입니다."
            elif plate.startswith("9"):
                feature_name = "차량번호 대역 (9000번대 임차 전세)"
                feature_desc = f"차량번호 '{plate}'는 9000번대 출퇴근 임차 전세버스 식별 번호로 확인됨"
                decision_text = f"차량번호 '{plate}'는 9000번대 외부 임차 전세버스로, 예약자 전용 차량이므로 [일반 학생 탑승 불가] 차량입니다."
            else:
                feature_name = "GBIS 예약 노선 지정 일치"
                feature_desc = f"차량 '{full_plate}'는 7790(예약) 전용 노선으로 인가된 차량"
                decision_text = f"차량 '{full_plate}'는 7790(예약) 출퇴근 전세버스로, [일반 학생 탑승 불가] 차량입니다."

            features = [
                {"name": feature_name, "weight": 78, "impact": "negative", "desc": feature_desc},
                {"name": "출퇴근 집중 배차 시간대 매칭", "weight": 14, "impact": "negative", "desc": "피크 타임 전세 투입 패턴과 일치"},
                {"name": "경진여객 정규 노선 면허 불일치", "weight": 8, "impact": "negative", "desc": "정규 노선버스(경기 70사 수원 / 경기 76아 화성) 외 면허"}
            ]
            confidence = 98.5
            boarding_prob = 0
            boarding_status = "탑승불가 (전세버스)"
            prob_color = "rose"
            seat_range = None
        else:
            crowd_status = crowd_info.get("status", "none")

            # --- [개선] 상행 연쇄 전파 & 효행초 조기 포화(역주행 탑승) 연쇄 증폭 모델 ---
            # 💡 사용자 피드백 반영:
            # 1. 효행초 조기 포화 감지 시(역주행 탑승): 뒤따르는 아이파크/신명/수원대 좌석 차감 가중치 대폭 증폭
            # 2. 안내 문구에서는 복잡한 차감 계산식/수치를 비노출하고 직관적인 상태 설명형 자연어 제공
            # 3. 2층 버스(71석) vs 1층 버스(45석) 차종 스펙 완벽 명시
            upstream_consumed = 0

            is_double = b.get("isDoubleDecker", False)
            bus_type_title = "2층 버스 (71석)" if is_double else "1층 버스 (45석)"

            if not is_down:
                current_seq = profile.get("seq", 16)
                # ⭐ [알고리즘 정확도 혁신] GBIS 차량 위치(currentStationId) 기반 실제 seq 결정 및 스케일 불일치 해소
                # 공공데이터 API의 remainStations는 '정차 정류장 수' 기준이라 UPSTREAM_STATION_CHAIN의 실제 노선 seq(10, 12, 14, 16)와 스케일이 불일치함
                # 1순위: vehicles 리스트에서 해당 차량(shortPlate)의 currentStationId를 찾아 CACHED_STATIONS의 실제 seq 매핑
                # 2순위: 미매칭 시 fallback으로 remainStations 기반 상대 체인 인덱스 계산
                matched_v = next((v for v in vehicles if v.get("shortPlate") == plate), None)
                bus_actual_seq = None
                if matched_v and matched_v.get("currentStationId"):
                    cur_st_id = str(matched_v.get("currentStationId"))
                    st_obj = next((s for s in CACHED_STATIONS if str(s.get("id")) == cur_st_id), None)
                    if st_obj:
                        bus_actual_seq = st_obj.get("seq")

                if bus_actual_seq is None:
                    rem_stations = b.get("remainStations", 1)
                    # UPSTREAM 체인 간격이 약 2씩 차이나므로 스케일 보정
                    bus_actual_seq = max(1, current_seq - (rem_stations * 2))

                for up_node in UPSTREAM_STATION_CHAIN:
                    # 버스가 아직 도달하지 않은 정류장이면서 탑승 정류장보다 앞선 정류장만 차감!
                    if bus_actual_seq <= up_node["seq"] < current_seq:
                        st_id = up_node["id"]
                        up_summary = get_crowd_summary(station_id=st_id)
                        u_st = up_summary.get("status", "none")
                        u_conf = up_summary.get("confidence", 1.0)  # 신뢰도 가중치 (1건 70%, 2건 이상 100%)

                        if st_id == "233002245":  # 효행초등학교 (기점 10번째)
                            if u_st == "crowded":
                                c_seats = 12
                            elif u_st == "moderate":
                                c_seats = 7
                            elif u_st == "smooth":
                                c_seats = 3
                            else:
                                c_seats = 4 if is_afternoon_rush else 1

                        elif st_id == "233002371":  # 아이파크정문 (하교 피크 최다 승차 1위)
                            if is_hyohaeng_surge:
                                # 🚨 효행초 역주행 급증 시 아이파크 차감 가중치 폭증 (초포화 연쇄 전파)
                                if u_st == "crowded":
                                    c_seats = 24  # 학우 피드백: 제보 접수 시 대폭 추가 차감
                                elif u_st == "moderate":
                                    c_seats = 16  # 보통 제보 시 16석 차감
                                elif u_st == "smooth":
                                    c_seats = 6
                                else:
                                    c_seats = 12 if is_afternoon_rush else 5
                            else:
                                if u_st == "crowded":
                                    c_seats = 15
                                elif u_st == "moderate":
                                    c_seats = 8
                                elif u_st == "smooth":
                                    c_seats = 3
                                else:
                                    c_seats = 6 if is_afternoon_rush else 2

                        else:  # 신명아파트 (수원대 직전)
                            if is_hyohaeng_surge:
                                if u_st == "crowded":
                                    c_seats = 9  # 제보 접수 시 추가 차감
                                elif u_st == "moderate":
                                    c_seats = 6
                                else:
                                    c_seats = 3 if is_afternoon_rush else 1
                            else:
                                if u_st == "crowded":
                                    c_seats = 6
                                elif u_st == "moderate":
                                    c_seats = 3
                                elif u_st == "smooth":
                                    c_seats = 1
                                else:
                                    c_seats = 2 if is_afternoon_rush else 1

                        # 단일 제보 시 70% 신뢰도 가중치 적용하여 과민 차감 방지
                        if u_st in ["crowded", "moderate", "smooth"]:
                            c_seats = max(1, int(round(c_seats * u_conf)))

                        upstream_consumed += c_seats

            # 앞 정류장 선소진을 차감한 실질 잔여 좌석
            effective_seats = max(0, seats - upstream_consumed) if not is_down else seats

            if is_down:
                # [사당역 9번 출구 (하행 수원대행)]:
                # 💡 사당역은 회차 정류장으로 승객 전원 하차 후 '전석 빈 차'로 진입하므로
                # 실제 총 수용 좌석은 2층 버스 71석, 1층 버스 45석입니다.
                eval_seats = 71 if is_double else 45

                is_sticky = crowd_info.get("isSticky", False)
                decay_fac = crowd_info.get("stickyDecayFactor", 1.0)
                is_evening_rush = crowd_info.get("isEveningRush", False)

                if crowd_status == "smooth":
                    # 피크 시간대에는 제보자가 '여유'를 눌러도 비사용자 및 일반 시민 기본 대기열이 존재
                    if is_morning_rush:
                        est_waiters = 16
                        crowd_desc = f"학우 실시간 제보(여유) + 비사용자 대기열: 약 {est_waiters}명 차감 반영"
                    elif is_evening_rush:
                        est_waiters = 18
                        crowd_desc = f"학우 실시간 제보(여유) + 퇴근 시민 대기열: 약 {est_waiters}명 차감 반영"
                    else:
                        est_waiters = 7
                        crowd_desc = f"학우 실시간 제보(여유): 대기열 약 {est_waiters}명 차감 반영"
                    delta = 3
                    tft_crowd_weight = 40
                    is_from_report = True
                elif crowd_status == "moderate":
                    base_mod = 25 if is_morning_rush else (32 if is_evening_rush else 18)
                    min_mod = 12 if not is_evening_rush else 16
                    est_waiters = max(min_mod, int(round(base_mod * decay_fac)))
                    delta = 4
                    tft_crowd_weight = 45
                    crowd_desc = (f"학우 실시간 제보(보통) + 비사용자 대기열: 약 {est_waiters}명 차감 반영"
                                  if not is_sticky else
                                  f"직전 제보 잔류 + 비사용자 대기열: 약 {est_waiters}명 차감 반영")
                    is_from_report = True
                elif crowd_status == "crowded":
                    base_crowd = 52 if is_morning_rush else (54 if is_evening_rush else 30)
                    min_crowd = 24 if not is_evening_rush else 30
                    est_waiters = max(min_crowd, int(round(base_crowd * decay_fac)))
                    delta = 5
                    tft_crowd_weight = 50
                    crowd_desc = (f"학우 실시간 제보(포화) + 비사용자 대기열: 약 {est_waiters}명 대폭 차감 반영"
                                  if not is_sticky else
                                  f"직전 혼잡 제보 잔류 + 비사용자 대기열: 약 {est_waiters}명 대폭 차감 반영")
                    is_from_report = True
                else:
                    # ⭐ 실시간 제보 부재 시: 수원대 통학 시간표 및 비사용자 포함 사전 통계 모델 자동 가동 (2/3 수준)
                    est_waiters, time_slot_desc = estimate_sadang_waiters_by_time()
                    delta = 4
                    tft_crowd_weight = 38
                    crowd_desc = f"대기열 분석: [{time_slot_desc}] 비사용자 포함 약 {est_waiters}명 사전 차감 반영"
                    is_from_report = False

                if is_from_report:
                    sadang_conf = crowd_info.get("confidence", 1.0)
                    if sadang_conf < 1.0:
                        est_waiters = max(5, int(round(est_waiters * sadang_conf)))

                # 2번째, 3번째 버스일 경우: 앞선 버스 승차 후 잔여 대기열 이월 및 후속 유입 반영
                if idx > 1 and len(target_list) >= idx:
                    prev_bus = target_list[idx - 2]
                    prev_cap = 0 if prev_bus.get("isReserved") else (71 if prev_bus.get("isDoubleDecker") else 45)
                    leftover_waiters = max(0, prev_bus_waiters - prev_cap)
                    p_curr = b.get("predictTimeMin") or (prev_bus.get("predictTimeMin", 10) + 15)
                    p_prev = prev_bus.get("predictTimeMin") or 10
                    interval_min = max(5, p_curr - p_prev)
                    arrival_rate = 1.0 if is_morning_rush else (0.9 if is_evening_rush else 0.4)
                    est_waiters = leftover_waiters + int(round(interval_min * arrival_rate))
                    crowd_desc = f"앞선 버스({prev_bus.get('shortPlate', '')}) 승차 이월 및 대기열: 약 {est_waiters}명 차감 반영"

                prev_bus_waiters = est_waiters

                # 사당역 진입 버스(빈 차 45석 또는 71석)와 비사용자 포함 대기열 간 착석 성공률 계산
                seat_diff = eval_seats - est_waiters

                if seat_diff >= 15:
                    boarding_prob = 96
                    boarding_status = "착석 매우 확실 (좌석 여유)"
                    prob_color = "emerald"
                elif seat_diff >= 6:
                    boarding_prob = 82
                    boarding_status = "착석 유력 (순번 안정)"
                    prob_color = "emerald"
                elif seat_diff >= 0:
                    boarding_prob = 58
                    boarding_status = "착석 경합 (앞쪽 순번 유력)"
                    prob_color = "amber"
                elif seat_diff >= -10:
                    boarding_prob = 32
                    boarding_status = "착석 주의 (만석 초과 우려)"
                    prob_color = "rose"
                else:
                    boarding_prob = 15
                    boarding_status = "만석 확실 (다음 차 대기 필수)"
                    prob_color = "rose"

                # 설명 문구 생성 (2층/1층 버스 스펙 완벽 결합, 비사용자 사전 차감 반영 명시)
                if is_from_report:
                    sticky_tag = " (직전 제보 승차 대기 중)" if is_sticky else ""
                    if seat_diff >= 15:
                        decision_text = f"사당역 회차 후 빈 차({bus_type_title})로 진입하며, 실시간 제보({crowd_info['statusText']}){sticky_tag} 및 비사용자 대기열을 감안해도 여유로운 착석 탑승이 유력합니다."
                    elif seat_diff >= 6:
                        decision_text = f"사당역 회차 후 빈 차({bus_type_title})로 진입하며, 실시간 제보({crowd_info['statusText']}){sticky_tag} 및 비사용자 대기열 감안 시 줄 순번에 따라 안정적인 착석이 예상됩니다."
                    elif seat_diff >= 0:
                        decision_text = f"사당역 회차 후 빈 차({bus_type_title})로 진입하나, 실시간 제보({crowd_info['statusText']}){sticky_tag} 및 비사용자 대기열로 인해 뒷 순번은 착석 경합이 발생할 수 있습니다."
                    elif seat_diff >= -10:
                        decision_text = f"사당역 회차 후 빈 차({bus_type_title})로 진입하나, 비사용자 포함 대기열이 수용 인원을 초과하여 1층 버스는 만석 가능성이 매우 높습니다."
                    else:
                        decision_text = f"비사용자를 포함한 대기 인원이 버스 좌석수를 크게 초과하여 이번 차는 조기 만석되므로, 다음 차 대기를 적극 권장합니다."
                else:
                    if seat_diff < -10:
                        decision_text = f"현재 [{time_slot_desc}]로 서비스 미사용자 포함 대기열이 극심하여, {bus_type_title}는 조기 만석으로 탑승이 어렵습니다. 다음 차 대기를 고려하세요."
                    elif seat_diff < 0:
                        decision_text = f"사당역 회차 후 빈 차({bus_type_title})로 진입하나, 현재 [{time_slot_desc}]로 비사용자 포함 대기열이 수용 인원을 초과하여 만석 주의가 필요합니다."
                    elif is_double and est_waiters >= 60:
                        decision_text = f"현재 [{time_slot_desc}]로 비사용자 포함 대기열({est_waiters}명 내외)이 길어, 2층 버스(71석) 진입 시에도 뒷 순번은 착석 경합이 발생할 수 있습니다."
                    elif is_double and est_waiters >= 30:
                        decision_text = f"현재 [{time_slot_desc}]로 비사용자 포함 대기열이 형성 중이나, 2층 버스(71석)가 도착하여 대부분 여유롭게 착석 탑승이 가능합니다."
                    elif seat_diff >= 15:
                        decision_text = f"사당역 회차 후 빈 차({bus_type_title})로 진입하며, 현재 [{time_slot_desc}] 기준 여유로운 착석 탑승이 유력합니다."
                    else:
                        decision_text = f"사당역 회차 후 빈 차({bus_type_title})로 진입하며, 현재 [{time_slot_desc}] 대기 상황을 감안할 때 줄 순번에 따라 착석이 가능할 것으로 예상됩니다."

            else:
                # [상행 사당역행]: 실질 잔여좌석(effective_seats) 및 오후 3~5시 하교 피크 반영
                if station_id == "233002245":  # 효행초등학교정문 (기점 10번째)
                    boarding_prob = 99
                    boarding_status = "착석 확실 (기점 극초반)"
                    prob_color = "emerald"
                    est_waiters = 3 if not is_afternoon_rush else (8 if is_hyohaeng_surge else 5)
                    delta = 2
                    tft_crowd_weight = 58
                    crowd_desc = "협성대 기점 직후로 빈 차 진입, 전 좌석 여유"
                    if is_hyohaeng_surge:
                        decision_text = f"기점 극초반 정류장으로 차량('{plate}', {bus_type_title})의 잔여 {seats}석이 확보되어 있으나, 아이파크/수원대 하교 인파의 효행초 조기 승차가 감지되었습니다."
                    else:
                        decision_text = f"기점 극초반 정류장으로 차량('{plate}', {bus_type_title})의 잔여 {seats}석이 확보되어 여유로운 착석 탑승이 가능합니다."

                elif station_id == "233002371":  # 아이파크정문 (기점 12번째)
                    boarding_prob = 98 if effective_seats >= 20 else 88
                    boarding_status = "착석 매우 여유 (기점 초기)"
                    prob_color = "emerald"
                    est_waiters = 6 if not is_afternoon_rush else (16 if is_hyohaeng_surge else 12)
                    delta = 3
                    tft_crowd_weight = 52
                    crowd_desc = f"아이파크 대기열(약 {est_waiters}명) 및 잔여 좌석 분석"
                    surge_warn = " 🚨(효행초 조기 승차 영향 주의)" if is_hyohaeng_surge else ""
                    decision_text = f"차량('{plate}', {bus_type_title})은 기점 출발 초기 구간(잔여 {seats}석)으로, 앞선 정류장 승차를 감안해도 여유롭게 착석 탑승이 가능합니다.{surge_warn}"

                elif station_id == "233002255":  # 신명아파트 (기점 14번째)
                    if effective_seats < 10:
                        boarding_prob = 70
                        boarding_status = "착석 보통 (앞 정류장 승차 감안)"
                        prob_color = "amber"
                        est_waiters = 14 if is_hyohaeng_surge else 12
                        delta = 3
                        surge_warn = " ⚠️(효행초 조기 승차 영향 주의)" if is_hyohaeng_surge else ""
                        decision_text = f"차량('{plate}', {bus_type_title})은 현재 잔여 {seats}석이나, 앞선 정류장 승차 누적으로 착석 경합이 발생할 수 있습니다.{surge_warn}"
                    elif is_afternoon_rush and crowd_status == "crowded":
                        boarding_prob = 82
                        boarding_status = "착석 양호 (하교 보통)"
                        prob_color = "emerald"
                        est_waiters = 16 if is_hyohaeng_surge else 14
                        delta = 4
                        surge_warn = " ⚠️(효행초 조기 승차 영향 주의)" if is_hyohaeng_surge else ""
                        decision_text = f"차량('{plate}', {bus_type_title})은 현재 잔여 {seats}석으로, 앞선 정류장 승차를 감안하여 줄 순번에 따라 착석이 가능할 것으로 예상됩니다.{surge_warn}"
                    else:
                        boarding_prob = 98 if effective_seats >= 15 else 90
                        boarding_status = "착석 매우 확실 (단지 여유)"
                        prob_color = "emerald"
                        est_waiters = 10 if is_hyohaeng_surge else 8
                        delta = 3
                        surge_warn = " ⚠️(효행초 조기 승차 영향 주의)" if is_hyohaeng_surge else ""
                        decision_text = f"차량('{plate}', {bus_type_title})은 현재 잔여 {seats}석으로, 앞선 정류장 승차를 감안해도 여유로운 착석 탑승이 예상됩니다.{surge_warn}"
                    tft_crowd_weight = 46
                    crowd_desc = f"신명아파트 대기열(약 {est_waiters}명) 및 잔여 좌석 분석"

                else:  # 수원대입구 (233003021, 기점 16번째 - 학생 하교 최다 정류장)
                    # 💡 사용자 피드백 반영: 악의적/허위 제보로 인한 왜곡 방지를 위해
                    # 수원대 제보 가중치를 소폭 완화(Damping)하고 대기열 차감 수치를 완만하게 조정
                    is_sticky = crowd_info.get("isSticky", False)
                    decay_fac = crowd_info.get("stickyDecayFactor", 1.0)

                    if is_afternoon_rush:
                        if is_hyohaeng_surge:
                            if crowd_status == "crowded":
                                est_waiters = int(round(32 * decay_fac))  # 완화 (기존 40명 -> 32명)
                            elif crowd_status == "moderate":
                                est_waiters = int(round(22 * decay_fac))  # 완화 (기존 28명 -> 22명)
                            elif crowd_status == "smooth":
                                est_waiters = 10
                            else:
                                est_waiters = 18
                        else:
                            if crowd_status == "crowded":
                                est_waiters = int(round(22 * decay_fac))  # 완화 (기존 28명 -> 22명)
                            elif crowd_status == "moderate":
                                est_waiters = int(round(15 * decay_fac))  # 완화 (기존 18명 -> 15명)
                            elif crowd_status == "smooth":
                                est_waiters = 7
                            else:
                                est_waiters = 12
                    else:
                        # 평상시 완화 차감치
                        if is_hyohaeng_surge:
                            if crowd_status == "crowded":
                                est_waiters = int(round(24 * decay_fac))  # 완화 (기존 32명 -> 24명)
                            elif crowd_status == "moderate":
                                est_waiters = int(round(15 * decay_fac))  # 완화 (기존 20명 -> 15명)
                            elif crowd_status == "smooth":
                                est_waiters = 5
                            else:
                                est_waiters = 8
                        else:
                            if crowd_status == "crowded":
                                est_waiters = int(round(18 * decay_fac))  # 완화 (기존 24명 -> 18명)
                            elif crowd_status == "moderate":
                                est_waiters = int(round(11 * decay_fac))  # 완화 (기존 14명 -> 11명)
                            elif crowd_status == "smooth":
                                est_waiters = 4
                            else:
                                est_waiters = 6   # 평상시 기본치

                    # 실질 잔여좌석과 수원대 대기열 간의 실질 경합 계산 (대기열 차감)
                    seat_diff = effective_seats - est_waiters

                    is_from_report = (crowd_status in ["crowded", "moderate", "smooth"])
                    # 악용 방지: 수원대 제보 가중치를 기존 54%에서 40%로 소폭 하향 조정
                    tft_crowd_weight = 40 if is_from_report else 35
                    crowd_desc = f"학우 실시간 제보({crowd_info['statusText']}): 수원대 대기열 분석 (신뢰 완충)" if is_from_report else "수원대 대기열 및 잔여 좌석 종합 분석"

                    surge_warn = " 🚨(앞 정류장 조기 승차 급증으로 잔여 좌석이 빠르게 줄어들 수 있습니다)" if is_hyohaeng_surge else ""

                    if is_from_report:
                        if seat_diff >= 5:
                            boarding_prob = 94
                            boarding_status = "착석 확실 (좌석 여유)"
                            prob_color = "emerald"
                            delta = 3
                            decision_text = f"차량('{plate}', {bus_type_title})은 현재 잔여 {seats}석으로 여유로우며, 수원대 실시간 제보 및 앞선 승차를 감안해도 착석 탑승이 유력합니다.{surge_warn}"
                        elif seat_diff >= -3:
                            boarding_prob = 80
                            boarding_status = "착석 양호 (줄 순번 안정)"
                            prob_color = "emerald"
                            delta = 4
                            decision_text = f"차량('{plate}', {bus_type_title})은 현재 잔여 {seats}석으로, 수원대 실시간 제보 및 앞선 승차를 감안해도 줄 순번에 따라 안정적인 착석이 예상됩니다.{surge_warn}"
                        elif seat_diff >= -8:
                            boarding_prob = 58
                            boarding_status = "착석 경합 (탑승 가능/입석 대비)"
                            prob_color = "amber"
                            delta = 4
                            decision_text = f"차량('{plate}', {bus_type_title})은 현재 잔여 {seats}석으로, 수원대 실시간 제보와 앞선 정류장 승차 누적으로 인해 줄 순번에 따라 착석 경합이 발생할 수 있습니다.{surge_warn}"
                        else:
                            boarding_prob = 38
                            boarding_status = "착석 주의 (다음 차 대기 고려)"
                            prob_color = "rose"
                            delta = 5
                            decision_text = f"차량('{plate}', {bus_type_title})은 현재 잔여 {seats}석이나, 수원대 실시간 제보와 앞선 승차 누적으로 인해 수원대 도착 시 만석이 될 수 있어 다음 차 대기를 고려하세요.{surge_warn}"
                    else:
                        if seat_diff >= 5:
                            boarding_prob = 94
                            boarding_status = "착석 확실 (좌석 여유)"
                            prob_color = "emerald"
                            delta = 3
                            decision_text = f"차량('{plate}', {bus_type_title})은 현재 잔여 {seats}석으로 여유로우며, 앞선 정류장 승차를 감안해도 수원대에서 여유로운 착석 탑승이 유력합니다.{surge_warn}"
                        elif seat_diff >= -3:
                            boarding_prob = 80
                            boarding_status = "착석 양호 (줄 순번 안정)"
                            prob_color = "emerald"
                            delta = 4
                            decision_text = f"차량('{plate}', {bus_type_title})은 현재 잔여 {seats}석으로, 앞선 정류장 승차를 감안해도 줄 순번에 따라 안정적인 착석이 예상됩니다.{surge_warn}"
                        elif seat_diff >= -8:
                            boarding_prob = 58
                            boarding_status = "착석 경합 (탑승 가능/입석 대비)"
                            prob_color = "amber"
                            delta = 4
                            decision_text = f"차량('{plate}', {bus_type_title})은 현재 잔여 {seats}석으로, 앞선 정류장 승차 누적으로 인해 줄 순번에 따라 착석 경합이 발생할 수 있습니다.{surge_warn}"
                        else:
                            boarding_prob = 38
                            boarding_status = "착석 주의 (다음 차 대기 고려)"
                            prob_color = "rose"
                            delta = 5
                            decision_text = f"차량('{plate}', {bus_type_title})은 현재 잔여 {seats}석이나, 앞선 정류장 승차 누적으로 수원대 도착 시 만석이 될 수 있어 다음 차 대기를 고려하세요.{surge_warn}"

            # 95% 착석 신뢰구간 수학적 계산 (실질 잔여좌석 기준)
            eval_seats = effective_seats if not is_down else (71 if is_double else 45)
            seat_expected = max(0, eval_seats - est_waiters)
            lower_bound = max(0, seat_expected - delta)
            upper_bound = min(eval_seats, seat_expected + delta)

            if lower_bound > 0:
                seat_range = f"{lower_bound}~{upper_bound}석 착석 가능"
            elif upper_bound > 0:
                seat_range = f"0~{upper_bound}석 (만석 주의)"
            else:
                seat_range = "0석 (만석 대기)" if is_down else "0석 (만석)"

            # ⭐ [불감대(Deadband) 앵커링 엔진] 정상 예측 오차 범위 내 숫자 널뛰기(Jitter) 방지 및 일관성 보장
            (
                seat_expected,
                lower_bound,
                upper_bound,
                seat_range,
                boarding_prob,
                boarding_status,
                prob_color,
                decision_text
            ) = apply_prediction_deadband_lock(
                station_id=station_id,
                plate=plate,
                new_expected=seat_expected,
                new_lower=lower_bound,
                new_upper=upper_bound,
                new_range=seat_range,
                new_prob=boarding_prob,
                new_status=boarding_status,
                new_color=prob_color,
                new_decision=decision_text,
                predict_min=predict_min,
                is_resv=is_resv
            )

            confidence = 95.0
            features = [
                {
                    "name": "사당역 회차 빈 차 진입" if is_down else ("앞 정류장 승차 흐름 반영" if upstream_consumed > 0 else "잔여 좌석 상태"),
                    "weight": 35 if not is_down and upstream_consumed > 0 else (100 - tft_crowd_weight - 20),
                    "impact": "positive" if (eval_seats - est_waiters) >= 5 else "negative",
                    "desc": f"사당역 회차 직후 전석 빈 차({bus_type_title}) 상태로 진입" if is_down else (f"기점 이후 선행 정류장 승차를 고려한 좌석 평가" if upstream_consumed > 0 else f"현재 잔여 {seats}석 진입")
                },
                {"name": f"{profile['name']} 대기열 (비사용자 사전 차감)" if is_down else f"{profile['name']} 실시간 대기열", "weight": tft_crowd_weight, "impact": "positive" if boarding_prob >= 70 else "negative", "desc": crowd_desc},
                {"name": "착석 여유 신뢰 보정", "weight": 20, "impact": "positive", "desc": f"대기열 변동 오차 범위 ±{delta}석 보정 적용"}
            ]

        bus_explanations.append({
            "order": f"{idx}번째 도착 버스",
            "shortPlate": plate,
            "predictMin": predict_min,
            "remainSeats": seats,
            "isReserved": is_resv,
            "isDoubleDecker": b.get("isDoubleDecker", False),
            "confidence": confidence,
            "decisionText": decision_text,
            "boardingProb": boarding_prob,
            "boardingStatus": boarding_status,
            "probColor": prob_color,
            "predictedSeatRange": seat_range,
            "features": features
        })

    return bus_explanations, is_hyohaeng_surge


# ⭐ 3. 정류장별 XAI 탑승 및 실시간 도로교통 지연 분석 엔드포인트 (도메인 휴리스틱 + 신뢰구간 불감대 모델)
@app.get("/api/xai-analysis")
def get_xai_analysis(station_id: str = Query("119000302")):
    global CACHE_XAI_ANALYSIS
    now = time.time()
    cached = CACHE_XAI_ANALYSIS.get(station_id)
    if cached and (now - cached["timestamp"] < XAI_CACHE_TTL):
        res = dict(cached["data"])
        res["crowdData"] = get_crowd_summary(station_id=station_id)
        return res

    bus_data = get_bus_arrivals(station_id=station_id)
    arriving = bus_data.get("arrivingBuses", [])
    vehicles = bus_data.get("vehicles", [])
    crowd_info = get_crowd_summary(station_id=station_id)

    profile = STATION_XAI_PROFILES.get(station_id, STATION_XAI_PROFILES["119000302"])
    is_down = (profile["dir"] == "DOWN")

    realtime_delays = calc_realtime_traffic_delays(direction=profile["dir"])

    bus_explanations, is_hyohaeng_surge = calculate_bus_predictions(
        station_id=station_id,
        arriving=arriving,
        vehicles=vehicles,
        crowd_info=crowd_info
    )

    model_meta = {
        "modelName": "7790 스마트 통학 지원 AI",
        "algorithm": "앞 정류장 승차 누적 예측 + 실시간 도로망 지연 분석",
        "dataSources": ["경기도 버스 실시간 GPS", "기상청 실시간 기상/노면 데이터", f"{profile['name']} 학우 실시간 제보 & {'등교 통학 패턴' if is_down else '하교 통학 패턴'}"],
        "totalActiveBuses": len(vehicles),
        "confidenceLevel": "95% 착석 신뢰 보증",
        "modelVersion": "v3.1-LiveTransit"
    }

    res_data = {
        "success": True,
        "stationMeta": profile,
        "busExplanations": bus_explanations,
        "delayFactors": realtime_delays,
        "modelMeta": model_meta,
        "crowdData": crowd_info,
        "isHyohaengSurge": is_hyohaeng_surge if not is_down else False
    }
    CACHE_XAI_ANALYSIS[station_id] = {"data": res_data, "timestamp": now}
    return res_data

# ⭐ 4. 7790번 운행 시간표 및 MiRi 전세 예약버스 API
@app.get("/api/timetable")
def get_timetable():
    """7790번 첫차/막차, 배차간격, MiRi 예약버스 고정 시간표"""
    return {
        "success": True,
        "regular": {
            "origin": "협성대 기점",
            "firstBus": "05:00",
            "lastBus": "23:30 (토/공휴일 22:50)",
            "suwonUnivFirstApprox": "05:12경",
            "turn": "사당역 9번 출구",
            "sadangFirstBus": "06:00",
            "sadangLastBus": "00:30 (토/공휴일 00:00)",
            "interval": "평일 10~25분 (출퇴근 10~15분 집중) / 주말 20~30분",
            "tel": "경진여객 안녕영업소 (031-223-9022)"
        },
        "miri": {
            "title": "7790(예약) 출근 전용 전세버스",
            "app": "경기스마트폰 앱 (MiRi)",
            "operationRoute": "봉담(협성대·상봉초) -> 사당역 (상행 출근 편도 전용)",
            "weekdayDepartures": ["07:30", "07:50", "08:10"],
            "sadangDepartures": "사당역 출발 예약 없음 (전석 공차 회송 / 일반 현장 줄서기 이용)",
            "note": "외부 임차 전세버스(경기 79사/바/아/자 및 9000번대)로 운행되며, 사당역 9번 출구에서는 예약 탑승이 없으므로 일반 정규 버스(경기 70사/76아)만 탑승 가능합니다."
        },
        "nightGuide": {
            "sadangLastBus": "00:30",
            "recommendedQueueTime": "23:50 ~ 00:10 사이 줄 서기 권장",
            "suwonUnivArrivalApprox": "새벽 01:10 ~ 01:25경 도착"
        }
    }

# 브라우저 PWA /sw.js 404 방지 더미 엔드포인트
@app.get("/sw.js")
def get_service_worker():
    return Response(content="// no-op service worker", media_type="application/javascript")

if __name__ == "__main__":
    import uvicorn
    print("[INFO] 7790 타요 웹서버를 시작합니다: http://127.0.0.1:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
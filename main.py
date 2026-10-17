import os
import re
import datetime
import time
import json
import threading
from typing import Literal, Tuple
import requests
import urllib3
from urllib.parse import quote
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from core.config import (
    API_KEY,
    ADMIN_API_TOKEN,
    REMOTE_SYNC_URL,
    KST,
    get_kst_now,
    STATION_ID,
    ROUTE_REGULAR_ID,
    ROUTE_RESERVE_ID,
    BUS_CACHE_TTL,
    XAI_CACHE_TTL,
    TRAFFIC_CACHE_TTL,
    WEATHER_CACHE_TTL,
    GBIS_ROUTE_CACHE_TTL,
    GBIS_HEADERS,
    STATION_XAI_PROFILES,
    UPSTREAM_STATION_CHAIN,
    TIMETABLE_INFO
)
from services.gbis_client import (
    HTTP_SESSION,
    is_charter_or_reserved_bus,
    fetch_gbis_route_cached,
    parse_gbis_realtime_vehicles,
    fetch_full_route_stations,
    calc_realtime_traffic_delays
)
from services.online_learner import online_learner, is_friday_night_window
from services.telemetry_logger import process_live_vehicles_for_learning, merge_telemetry_csv

FILE_IO_LOCK = threading.Lock()
_LOCK_BUS_ARRIVALS = threading.Lock()
_LOCK_XAI_ANALYSIS = threading.Lock()
_ADMIN_ACTION_LOCK = threading.Lock()
_LAST_ADMIN_ACTION_TS = 0.0

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_bg_learning_started = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global CACHED_STATIONS, _bg_learning_started
    CACHED_STATIONS = fetch_full_route_stations()
    load_crowd_data_from_disk()
    if not _bg_learning_started:
        _bg_learning_started = True
        t = threading.Thread(target=_background_telemetry_learning_loop, daemon=True)
        t.start()
        print("[OK] 전 요일·전 시간대 실시간 버스 온라인 학습(Online-ML) 백그라운드 데몬 가동 시작!")
    yield

app = FastAPI(title="7790 & Smart Transit Router", lifespan=lifespan)

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

CACHE_BUS_ARRIVALS = {}      
CACHE_XAI_ANALYSIS = {}     
CROWD_REPORT_IP_RATE_LIMIT = {}  


PREDICTION_LOCK_REGISTRY = {}  # {f"{station_id}_{plate}": dict}

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

    global PREDICTION_LOCK_REGISTRY
    if is_resv or not plate:
        return new_expected, new_lower, new_upper, new_range, new_prob, new_status, new_color, new_decision

    now = time.time()
    lock_key = f"{station_id}_{plate}"

    if predict_min <= 1:
        PREDICTION_LOCK_REGISTRY.pop(lock_key, None)
        return new_expected, new_lower, new_upper, new_range, new_prob, new_status, new_color, new_decision

    expired_keys = [k for k, v in list(PREDICTION_LOCK_REGISTRY.items()) if now - v.get("timestamp", 0) > 1200]
    for k in expired_keys:
        PREDICTION_LOCK_REGISTRY.pop(k, None)

    record = PREDICTION_LOCK_REGISTRY.get(lock_key)

    if record:
        locked_low = record["lower_bound"]
        locked_high = record["upper_bound"]

        if locked_low <= new_expected <= locked_high:
            record["timestamp"] = now 
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
            pass

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


def get_real_client_ip(request: Request) -> str:
    """리버스 프록시(Render 등) 환경을 고려하여 스푸핑을 방어한 실제 클라이언트 IP 추출"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # 클라이언트가 임의로 앞단에 추가한 위조 헤더를 배제하기 위해 프록시가 부여한 마지막 유효 홉 사용
        hops = [ip.strip() for ip in forwarded.split(",") if ip.strip()]
        if hops:
            candidate = hops[-1] if len(hops) > 1 else hops[0]
            if re.match(r"^[0-9a-fA-F:.]{3,45}$", candidate):
                return candidate
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        clean_real = real_ip.strip()
        if re.match(r"^[0-9a-fA-F:.]{3,45}$", clean_real):
            return clean_real
    return request.client.host if request.client else "unknown"


def _enrich_cached_bus_response(station_id: str, cached_data: dict) -> dict:
    """캐시된 버스 도착 응답에 실시간 제보 및 최신 XAI 좌석 예측을 동적 결합"""
    res = dict(cached_data)
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


@app.get("/")
def get_home():
    return FileResponse("static/index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/api/buses")
def get_bus_arrivals(station_id: str = Query("119000302")):
    global CACHED_STATIONS, CACHE_BUS_ARRIVALS
    now = time.time()

    cached = CACHE_BUS_ARRIVALS.get(station_id)
    if cached and (now - cached["timestamp"] < BUS_CACHE_TTL):
        return _enrich_cached_bus_response(station_id, cached["data"])

    with _LOCK_BUS_ARRIVALS:
        now = time.time()
        cached = CACHE_BUS_ARRIVALS.get(station_id)
        if cached and (now - cached["timestamp"] < BUS_CACHE_TTL):
            return _enrich_cached_bus_response(station_id, cached["data"])

        if not CACHED_STATIONS:
            CACHED_STATIONS = fetch_full_route_stations()

        url_arrival = f"https://apis.data.go.kr/6410000/busarrivalservice/v2/getBusArrivalListv2?serviceKey={API_KEY}&stationId={station_id}&format=json"

        arriving_buses = []
        live_vehicles = []
        live_data = {}

        # 1. 사당역 도착 정보 (HTTP_SESSION 커넥션 풀 재사용)
        try:
            r_arr = HTTP_SESSION.get(url_arrival, timeout=4)
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

        try:
            live_data = fetch_gbis_route_cached()
            live_vehicles = parse_gbis_realtime_vehicles(live_data)
        except Exception as e:
            print(f"차량 API 오류: {e}")

        for ab in arriving_buses:
            short = ab.get("shortPlate")
            matched = next((v for v in live_vehicles if v.get("shortPlate") == short), None)
            if matched:
                ab["isDoubleDecker"] = matched.get("isDoubleDecker", False)
            elif "isDoubleDecker" not in ab:
                ab["isDoubleDecker"] = False

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
                    arriving_buses.sort(key=lambda x: x["predictTimeMin"])
            except Exception as e_fb:
                print(f"GBIS 실시간 3대 추적/폴백 산출 오류: {e_fb}")

        crowd_summary_data = get_crowd_summary(station_id=station_id)

        if live_vehicles:
            try:
                sadang_c_status = (
                    crowd_summary_data.get("status", "none")
                    if str(station_id) == "119000302"
                    else get_crowd_summary(station_id="119000302").get("status", "none")
                )
                process_live_vehicles_for_learning(
                    live_vehicles=live_vehicles,
                    now_dt=get_kst_now(),
                    heuristic_prior_fn=lambda: estimate_sadang_waiters_by_time(apply_online_ml=False),
                    crowd_status=sadang_c_status
                )
            except Exception as e_learn:
                print(f"실시간 차량 온라인 학습 트리거 오류: {e_learn}")

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
        CACHE_BUS_ARRIVALS[station_id] = {"data": res_data, "timestamp": time.time()}
        return res_data


def _background_telemetry_learning_loop():
    time.sleep(5)
    while True:
        try:
            now_dt = get_kst_now()
            if now_dt.hour >= 5 or now_dt.hour < 2:
                live_data = fetch_gbis_route_cached()
                parsed_vehicles = parse_gbis_realtime_vehicles(live_data)
                if parsed_vehicles:
                    sadang_c = get_crowd_summary(station_id="119000302").get("status", "none")
                    process_live_vehicles_for_learning(
                        live_vehicles=parsed_vehicles,
                        now_dt=now_dt,
                        heuristic_prior_fn=lambda: estimate_sadang_waiters_by_time(apply_online_ml=False),
                        crowd_status=sadang_c
                    )
        except Exception as e_bg:
            print(f"[WARN] 백그라운드 온라인 학습 루프 오류: {e_bg}")
        time.sleep(25)

CROWD_DATA_FILE = "crowd_reports_cache.json"
CROWD_REPORTS_BY_STATION = {}
CROWD_REPORT_IP_COOLDOWN = {}  # {ip: timestamp}
CROWD_STATE_LOCK = threading.Lock()

def save_crowd_data_to_disk():
    """원자적(Atomic) 파일 쓰기로 제보 데이터 디스크 보존 (데이터 손상 원천 방지)"""
    try:
        now = time.time()
        with CROWD_STATE_LOCK:
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
            tmp_file = CROWD_DATA_FILE + ".tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump({"reports": clean_reports, "ips": clean_ips}, f, ensure_ascii=False)
            os.replace(tmp_file, CROWD_DATA_FILE)
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
        clean_ips = {}
        for ip, record in data.get("ips", {}).items():
            t = record.get("time", 0) if isinstance(record, dict) else record
            dur = record.get("duration", 600) if isinstance(record, dict) else 600
            if now - t < dur:
                clean_ips[ip] = record
        with CROWD_STATE_LOCK:
            CROWD_REPORTS_BY_STATION = clean_reports
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
    if target_eta_min <= 1:
        dynamic_cooldown = 120 
    else:
        dynamic_cooldown = max(120, min(target_eta_min * 60, 600))  
    client_ip = get_real_client_ip(request)
    raw_uuid = request.headers.get("x-client-id", "").strip()
    client_uuid = raw_uuid if re.match(r"^[a-zA-Z0-9_-]{8,64}$", raw_uuid) else ""
    user_id = f"{client_ip}:{client_uuid}" if client_uuid else client_ip
    cooldown_key = f"{user_id}_{station_id}"
    ip_burst_key = f"ipburst_{client_ip}_{station_id}"

    with CROWD_STATE_LOCK:
        expired_ips = [ip for ip, hist in list(CROWD_REPORT_IP_RATE_LIMIT.items()) if not hist or (now - hist[-1] >= 60)]
        for ip in expired_ips:
            CROWD_REPORT_IP_RATE_LIMIT.pop(ip, None)
        ip_history = CROWD_REPORT_IP_RATE_LIMIT.get(client_ip, [])
        ip_history = [t for t in ip_history if now - t < 60]
        if len(ip_history) >= 8:
            return {
                "success": False,
                "message": "교내 네트워크에서 일시적으로 제보가 집중되었습니다. 1분 후 다시 시도해 주세요.",
                "remainCooldown": 30,
                "stationId": station_id
            }

        # 동일 IP에서 x-client-id를 랜덤 변조하여 연속 호출하는 스크립트 어뷰징 방어 (정류장당 최소 15초 간격)
        ip_burst_rec = CROWD_REPORT_IP_COOLDOWN.get(ip_burst_key)
        if isinstance(ip_burst_rec, dict) and (now - ip_burst_rec.get("time", 0) < 15):
            wait_sec = max(1, int(15 - (now - ip_burst_rec.get("time", 0))))
            return {
                "success": False,
                "message": f"동일 네트워크 연속 제보 보호 중입니다. {wait_sec}초 후 다시 시도해 주세요.",
                "remainCooldown": wait_sec,
                "stationId": station_id
            }

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

        CROWD_REPORTS_BY_STATION[station_id] = [r for r in CROWD_REPORTS_BY_STATION[station_id] if now - r["time"] < 600]
        CROWD_REPORTS_BY_STATION[station_id].append({
            "status": status,
            "time": now,
            "targetPlate": target_bus_plate,
            "targetEta": target_eta_min
        })
        CROWD_REPORT_IP_COOLDOWN[cooldown_key] = {"time": now, "duration": dynamic_cooldown}
        CROWD_REPORT_IP_COOLDOWN[ip_burst_key] = {"time": now, "duration": 15}

    save_crowd_data_to_disk()

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
    with CROWD_STATE_LOCK:
        reports = list(CROWD_REPORTS_BY_STATION.get(station_id, []))

    fresh_reports = [r for r in reports if now - r["time"] < 300]
    older_reports = [r for r in reports if 300 <= (now - r["time"]) < 600]
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
        bus_not_arrived_yet = (target_plate and target_plate == curr_first_plate) or (curr_first_min > 1)

        if last_r["status"] in ["crowded", "moderate"] and bus_not_arrived_yet:
            is_sticky = True
            valid_reports = [last_r]
            if age_sec < 450:     
                sticky_decay_factor = 0.85
            else:                  
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

    now_dt = get_kst_now()
    time_float = now_dt.hour + now_dt.minute / 60.0
    weekday = now_dt.weekday()  # 0:월 ~ 4:금, 5:토, 6:일

    if is_down:
        is_morning_peak = (weekday < 5) and (7.5 <= time_float <= (10.0 + 10.0/60.0))
        is_evening_peak = (weekday < 5) and (16.5 <= time_float <= 19.5)
        is_friday_night = is_friday_night_window(weekday, time_float)
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
        is_friday_night = False

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
            "isFridayNight": (is_down and is_friday_night),
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
            "isFridayNight": (is_down and is_friday_night),
            "isAfternoonRush": (is_peak and not is_down),
            "isSticky": False,
            "stickyDecayFactor": 1.0
        }

def estimate_sadang_waiters_by_time(apply_online_ml: bool = True, is_double_decker: bool = False):
    now_dt = get_kst_now()
    hour = now_dt.hour
    minute = now_dt.minute
    weekday = now_dt.weekday() 
    time_float = hour + minute / 60.0

    if weekday == 5 and time_float <= 0.67:
        base = 55
        est = int(round(base * 1.15)) 
        desc = "사당역 00:30 막차 귀가 시간대 (지하철 환승 및 귀가객 집중)"
        if apply_online_ml:
            est, desc, _ = online_learner.predict_sadang_waiters(now_dt, est, desc, is_double_decker=is_double_decker)
        return est, desc

    if weekday in [1, 2, 3, 4] and time_float <= 0.65:
        est = 25
        desc = "사당역 00:30 심야 막차 귀가 시간대 (비사용자 집중)"
        if apply_online_ml:
            est, desc, _ = online_learner.predict_sadang_waiters(now_dt, est, desc, is_double_decker=is_double_decker)
        return est, desc

    if weekday >= 5:
        if 11 <= hour <= 19:
            est, desc = 9, "주말 일반 통행 분산 (일반 승객 포함)"
        elif 21 <= hour <= 23:
            est, desc = 16, "주말 야간 귀가 시간대 (서울 나들이 귀가객 포함)"
        else:
            est, desc = 4, "주말 한산 시간대"
        if apply_online_ml:
            est, desc, _ = online_learner.predict_sadang_waiters(now_dt, est, desc, is_double_decker=is_double_decker)
        return est, desc

    if weekday == 4:
        if time_float >= 21.5:
            day_multiplier = 1.15
        elif time_float < 12.0:
            day_multiplier = 0.70
        else:
            day_multiplier = 1.0
    elif weekday in [0, 1]:
        day_multiplier = 1.05
    elif weekday == 3:
        day_multiplier = 0.90
    else:
        day_multiplier = 1.0

    day_name = ["월", "화", "수", "목", "금"][weekday]

    if 7.5 <= time_float < (7 + 50/60.0):  # 07:30 ~ 07:50 (1교시 등교열 급증 구간)
        base = 39
        desc = f"{day_name}요일 1교시 등교 급증 (비사용자·일반 승객 포함)"
    elif (7 + 50/60.0) <= time_float < (8 + 35/60.0):  # 07:50 ~ 08:35 (9시 1교시 등교 최대 극피크)
        base = 51
        desc = f"{day_name}요일 1교시 등교 극피크 (비사용자·일반 승객 포함)"
    elif (8 + 35/60.0) <= time_float < (8 + 55/60.0):  # 08:35 ~ 08:55 (1교시 직전 막바지 등교 & 2교시 유입)
        base = 35
        desc = f"{day_name}요일 1교시 막바지·2교시 유입 (비사용자·일반 승객 포함)"
    elif (8 + 55/60.0) <= time_float < (9 + 35/60.0):  # 08:55 ~ 09:35 (10:30 2교시 등교 피크)
        base = 31
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
    elif 19.5 <= time_float < 21.5:  # 19:30 ~ 21:30 (야간 귀가)
        base = 14
        desc = "야간 귀가 시간대 (일반 승객 포함)"
    elif 21.5 <= time_float < 22.5:  # 21:30 ~ 22:30 (불금 1차 귀가 급증 vs 평일 야간)
        if weekday == 4:
            base = 38
            desc = "야간 귀가 시간대 (서울 모임 귀가객 포함)"
        else:
            base = 12
            desc = "평일 야간 귀가 시간대 (일반 승객 포함)"
    elif 22.5 <= time_float < 23.75:  # 22:30 ~ 23:45
        if weekday == 4:
            base = 48
            desc = "심야 귀가 시간대 (귀가객 및 비사용자 포함)"
        else:
            base = 15
            desc = "평일 심야 귀가 시간대 (일반 승객 포함)"
    elif 23.75 <= time_float <= 24.0:  # 23:45 ~ 24:00
        if weekday == 4:
            base = 55
            desc = "사당역 00:30 막차 귀가 시간대 (지하철 환승 및 귀가객 집중)"
        else:
            base = 25
            desc = "사당역 00:30 막차 귀가 시간대 (비사용자 집중)"
    else:  # 심야/새벽
        base = 4
        desc = "심야/새벽 한산 시간대"

    est = int(round(base * day_multiplier))
    if apply_online_ml:
        est, desc, _ = online_learner.predict_sadang_waiters(now_dt, est, desc, is_double_decker=is_double_decker)
    return est, desc


def calculate_bus_predictions(station_id: str, arriving: list, vehicles: list, crowd_info: dict) -> tuple[list, bool]:
    profile = STATION_XAI_PROFILES.get(station_id, STATION_XAI_PROFILES["119000302"])
    is_down = (profile["dir"] == "DOWN")
    is_morning_rush = crowd_info.get("isMorningRush", False)
    is_afternoon_rush = crowd_info.get("isAfternoonRush", False)
    is_hyohaeng_surge = False
    if not is_down:
        hyohaeng_summary = get_crowd_summary(station_id="233002245")
        h_st = hyohaeng_summary.get("status", "none")
        if h_st in ["crowded", "moderate"]:
            is_hyohaeng_surge = True
        else:
            st_seq_map = {str(s["id"]): s.get("seq", 999) for s in CACHED_STATIONS if s.get("dir") == "UP"}
            for v in vehicles:
                if v.get("dir") == "U":
                    curr_st = str(v.get("currentStationId", ""))
                    v_seq = st_seq_map.get(curr_st, 999)
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
            upstream_consumed = 0

            is_double = b.get("isDoubleDecker", False)
            bus_type_title = "2층 버스 (71석)" if is_double else "1층 버스 (45석)"

            if not is_down:
                current_seq = profile.get("seq", 16)
                matched_v = next((v for v in vehicles if v.get("shortPlate") == plate), None)
                bus_actual_seq = None
                if matched_v and matched_v.get("currentStationId"):
                    cur_st_id = str(matched_v.get("currentStationId"))
                    st_obj = next((s for s in CACHED_STATIONS if str(s.get("id")) == cur_st_id), None)
                    if st_obj:
                        bus_actual_seq = st_obj.get("seq")

                if bus_actual_seq is None:
                    rem_stations = b.get("remainStations", 1)
                    bus_actual_seq = max(1, current_seq - (rem_stations * 2))

                for up_node in UPSTREAM_STATION_CHAIN:
                    if bus_actual_seq <= up_node["seq"] < current_seq:
                        st_id = up_node["id"]
                        up_summary = get_crowd_summary(station_id=st_id)
                        u_st = up_summary.get("status", "none")
                        u_conf = up_summary.get("confidence", 1.0)

                        if st_id == "233002245":  
                            if u_st == "crowded":
                                c_seats = 12
                            elif u_st == "moderate":
                                c_seats = 7
                            elif u_st == "smooth":
                                c_seats = 3
                            else:
                                c_seats = 4 if is_afternoon_rush else 1

                        elif st_id == "233002371": 
                            if is_hyohaeng_surge:
                                if u_st == "crowded":
                                    c_seats = 24 
                                elif u_st == "moderate":
                                    c_seats = 16 
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

                        else:  
                            if is_hyohaeng_surge:
                                if u_st == "crowded":
                                    c_seats = 9  
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

                        if u_st in ["crowded", "moderate", "smooth"]:
                            c_seats = max(1, int(round(c_seats * u_conf)))

                        upstream_consumed += c_seats

                if upstream_consumed > 0:
                    full_chain_default = sum(
                        (15 if node["id"] == "233002371" else (6 if node["id"] == "233002245" else 6))
                        for node in UPSTREAM_STATION_CHAIN if node["seq"] < current_seq
                    )
                    learned_full, _ = online_learner.get_upstream_learned_consumption(
                        station_id=station_id,
                        now_dt=get_kst_now(),
                        default_consumed=max(upstream_consumed, full_chain_default)
                    )
                    if full_chain_default > 0:
                        scale_ratio = min(1.0, upstream_consumed / float(full_chain_default))
                        upstream_consumed = max(1, int(round(learned_full * scale_ratio)))
                    else:
                        upstream_consumed = min(upstream_consumed, learned_full)

            effective_seats = max(0, seats - upstream_consumed) if not is_down else seats

            if is_down:
                eval_seats = 71 if is_double else 45

                is_sticky = crowd_info.get("isSticky", False)
                decay_fac = crowd_info.get("stickyDecayFactor", 1.0)
                is_evening_rush = crowd_info.get("isEveningRush", False)
                is_friday_night = crowd_info.get("isFridayNight", False)

                if crowd_status == "smooth":
                    if is_friday_night:
                        est_waiters = 22
                        crowd_desc = f"학우 실시간 제보(여유) + 심야 귀가객 대기열: 약 {est_waiters}명 차감 반영"
                    elif is_morning_rush:
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
                    base_mod = 42 if is_friday_night else (25 if is_morning_rush else (32 if is_evening_rush else 18))
                    min_mod = 22 if is_friday_night else (16 if is_evening_rush else 12)
                    est_waiters = max(min_mod, int(round(base_mod * decay_fac)))
                    delta = 4
                    tft_crowd_weight = 45
                    crowd_desc = (f"학우 실시간 제보(보통) + {'심야 귀가객' if is_friday_night else '비사용자'} 대기열: 약 {est_waiters}명 차감 반영"
                                  if not is_sticky else
                                  f"직전 제보 잔류 + {'심야 귀가객' if is_friday_night else '비사용자'} 대기열: 약 {est_waiters}명 차감 반영")
                    is_from_report = True
                elif crowd_status == "crowded":
                    base_crowd = 60 if is_friday_night else (52 if is_morning_rush else (54 if is_evening_rush else 30))
                    min_crowd = 34 if is_friday_night else (30 if is_evening_rush else 24)
                    est_waiters = max(min_crowd, int(round(base_crowd * decay_fac)))
                    delta = 5
                    tft_crowd_weight = 50
                    crowd_desc = (f"학우 실시간 제보(포화) + {'심야 막차 대기열' if is_friday_night else '비사용자 대기열'}: 약 {est_waiters}명 대폭 차감 반영"
                                  if not is_sticky else
                                  f"직전 혼잡 제보 잔류 + {'심야 막차 대기열' if is_friday_night else '비사용자'} 대기열: 약 {est_waiters}명 대폭 차감 반영")
                    is_from_report = True
                else:
                    est_waiters, time_slot_desc = estimate_sadang_waiters_by_time(
                        apply_online_ml=True,
                        is_double_decker=is_double
                    )
                    delta = 4
                    tft_crowd_weight = 38
                    crowd_desc = f"대기열 분석: [{time_slot_desc}] 비사용자 포함 약 {est_waiters}명 사전 차감 반영"
                    is_from_report = False

                if is_from_report:
                    sadang_conf = crowd_info.get("confidence", 1.0)
                    if sadang_conf < 1.0:
                        est_waiters = max(5, int(round(est_waiters * sadang_conf)))

                if idx > 1 and len(target_list) >= idx:
                    prev_bus = target_list[idx - 2]
                    prev_cap = 0 if prev_bus.get("isReserved") else (71 if prev_bus.get("isDoubleDecker") else 45)
                    leftover_waiters = max(0, prev_bus_waiters - prev_cap)
                    p_curr = b.get("predictTimeMin") or (prev_bus.get("predictTimeMin", 10) + 15)
                    p_prev = prev_bus.get("predictTimeMin") or 10
                    interval_min = max(5, p_curr - p_prev)
                    arrival_rate = 1.1 if is_friday_night else (1.0 if is_morning_rush else (0.9 if is_evening_rush else 0.4))
                    est_waiters = leftover_waiters + int(round(interval_min * arrival_rate))
                    crowd_desc = f"앞선 버스({prev_bus.get('shortPlate', '')}) 승차 이월 및 대기열: 약 {est_waiters}명 차감 반영"

                prev_bus_waiters = est_waiters

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
                    surge_warn = " (효행초 조기 승차 영향 주의)" if is_hyohaeng_surge else ""
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

                else: 
                    is_sticky = crowd_info.get("isSticky", False)
                    decay_fac = crowd_info.get("stickyDecayFactor", 1.0)

                    if is_afternoon_rush:
                        if is_hyohaeng_surge:
                            if crowd_status == "crowded":
                                est_waiters = int(round(32 * decay_fac)) 
                            elif crowd_status == "moderate":
                                est_waiters = int(round(22 * decay_fac)) 
                            elif crowd_status == "smooth":
                                est_waiters = 10
                            else:
                                est_waiters = 18
                        else:
                            if crowd_status == "crowded":
                                est_waiters = int(round(22 * decay_fac)) 
                            elif crowd_status == "moderate":
                                est_waiters = int(round(15 * decay_fac)) 
                            elif crowd_status == "smooth":
                                est_waiters = 7
                            else:
                                est_waiters = 12
                    else:
                        # 평상시 완화 차감치
                        if is_hyohaeng_surge:
                            if crowd_status == "crowded":
                                est_waiters = int(round(24 * decay_fac)) 
                            elif crowd_status == "moderate":
                                est_waiters = int(round(15 * decay_fac))  
                            elif crowd_status == "smooth":
                                est_waiters = 5
                            else:
                                est_waiters = 8
                        else:
                            if crowd_status == "crowded":
                                est_waiters = int(round(18 * decay_fac))  
                            elif crowd_status == "moderate":
                                est_waiters = int(round(11 * decay_fac))  
                            elif crowd_status == "smooth":
                                est_waiters = 4
                            else:
                                est_waiters = 6   

                    seat_diff = effective_seats - est_waiters

                    is_from_report = (crowd_status in ["crowded", "moderate", "smooth"])
                    tft_crowd_weight = 40 if is_from_report else 35
                    crowd_desc = f"학우 실시간 제보({crowd_info['statusText']}): 수원대 대기열 분석 (신뢰 완충)" if is_from_report else "수원대 대기열 및 잔여 좌석 종합 분석"

                    surge_warn = " (앞 정류장 조기 승차 급증으로 잔여 좌석이 빠르게 줄어들 수 있습니다)" if is_hyohaeng_surge else ""

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


@app.get("/api/xai-analysis")
def get_xai_analysis(station_id: str = Query("119000302")):
    global CACHE_XAI_ANALYSIS
    now = time.time()
    cached = CACHE_XAI_ANALYSIS.get(station_id)
    if cached and (now - cached["timestamp"] < XAI_CACHE_TTL):
        res = dict(cached["data"])
        res["crowdData"] = get_crowd_summary(station_id=station_id)
        res["onlineLearning"] = online_learner.get_status_summary()
        return res

    # _LOCK_BUS_ARRIVALS와의 교착(Deadlock) 방지를 위해 버스 데이터는 먼저 조회
    bus_data = get_bus_arrivals(station_id=station_id)

    with _LOCK_XAI_ANALYSIS:
        now = time.time()
        cached = CACHE_XAI_ANALYSIS.get(station_id)
        if cached and (now - cached["timestamp"] < XAI_CACHE_TTL):
            res = dict(cached["data"])
            res["crowdData"] = get_crowd_summary(station_id=station_id)
            res["onlineLearning"] = online_learner.get_status_summary()
            return res

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

        ml_summary = online_learner.get_status_summary()
        model_meta = {
            "modelName": "7790 스마트 통학 지원 AI",
            "algorithm": "전 요일·전 시간대 실시간 온라인 학습(Online-ML) + 불감대 좌석 예측",
            "dataSources": ["경기도 버스 실시간 GPS", "기상청 실시간 기상/노면 데이터", f"{profile['name']} 학우 실시간 제보 & {'등교·퇴근·심야 귀가 패턴' if is_down else '하교 통학 패턴'}"],
            "totalActiveBuses": len(vehicles),
            "confidenceLevel": "95% 착석 신뢰 보증",
            "modelVersion": "v3.2-OnlineML",
            "onlineLearning": ml_summary
        }

        res_data = {
            "success": True,
            "stationMeta": profile,
            "busExplanations": bus_explanations,
            "delayFactors": realtime_delays,
            "modelMeta": model_meta,
            "crowdData": crowd_info,
            "onlineLearning": ml_summary,
            "isHyohaengSurge": is_hyohaeng_surge if not is_down else False
        }
        CACHE_XAI_ANALYSIS[station_id] = {"data": res_data, "timestamp": time.time()}
        return res_data


@app.get("/api/ml-status")
def get_ml_status():
    """전 요일·전 시간대 실시간 온라인 머신러닝 학습 현황 조회 API"""
    return {
        "success": True,
        "onlineLearning": online_learner.get_status_summary()
    }


def _verify_admin_access(request: Request, token: str = "", min_interval_sec: float = 3.0) -> Tuple[bool, str]:
    """관리/동기화 엔드포인트 접근 제어 및 과부하(DoS) 방어 쿨다운 검증"""
    global _LAST_ADMIN_ACTION_TS
    client_ip = get_real_client_ip(request)
    is_local = client_ip in ("127.0.0.1", "::1", "localhost")

    if ADMIN_API_TOKEN and not is_local:
        header_token = request.headers.get("x-admin-token", "").strip()
        provided = header_token or (token or "").strip()
        if provided != ADMIN_API_TOKEN:
            return False, "관리자 인증 토큰(X-Admin-Token)이 유효하지 않습니다."

    with _ADMIN_ACTION_LOCK:
        now = time.time()
        if not is_local and (now - _LAST_ADMIN_ACTION_TS < min_interval_sec):
            return False, "요청이 너무 빈번합니다. 잠시 후 다시 시도해 주세요."
        _LAST_ADMIN_ACTION_TS = now
    return True, "ok"


@app.post("/api/ml-reload")
def reload_ml_model(request: Request, token: str = Query("")):
    """온라인 학습 모델 및 텔레메트리 최신 상태 다시 불러오기"""
    global CACHE_XAI_ANALYSIS
    ok, msg = _verify_admin_access(request, token=token, min_interval_sec=2.0)
    if not ok:
        return {"success": False, "error": msg}

    online_learner.load_state()
    CACHE_XAI_ANALYSIS.clear()
    return {"success": True, "onlineLearning": online_learner.get_status_summary()}


@app.post("/api/backup/sync-render")
def sync_from_render(request: Request, token: str = Query("")):
    """Render 배포 서버의 최신 텔레메트리 CSV 및 학습 데이터를 가져와 로컬과 병합 동기화"""
    global CACHE_XAI_ANALYSIS
    ok, msg = _verify_admin_access(request, token=token, min_interval_sec=5.0)
    if not ok:
        return {"success": False, "error": msg}

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        if ADMIN_API_TOKEN:
            headers["X-Admin-Token"] = ADMIN_API_TOKEN
        resp = HTTP_SESSION.get(f"{REMOTE_SYNC_URL.rstrip('/')}/api/backup/export", headers=headers, timeout=10)
        data = resp.json()

        remote_sadang = data.get("sadang_csv", "")
        if remote_sadang:
            merge_telemetry_csv("data/sadang_telemetry.csv", remote_sadang, lambda r: (r[0], r[5]), lambda r: r[0])

        remote_upstream = data.get("upstream_csv", "")
        if remote_upstream:
            merge_telemetry_csv("data/upstream_telemetry.csv", remote_upstream, lambda r: (r[0], r[1], r[6]), lambda r: r[0])

        online_learner.rebuild_from_csvs()
        online_learner.save_state()
        CACHE_XAI_ANALYSIS.clear()
        return {
            "success": True,
            "message": "Render 최신 학습 데이터 병합 동기화 완료",
            "summary": online_learner.get_status_summary()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/backup/import")
async def import_backup_data(request: Request, token: str = Query("")):
    """외부 백업 데이터(CSV 및 온라인 모델)를 현재 서버에 병합 복원 (Render 재배포 초기화 복구용)"""
    global CACHE_XAI_ANALYSIS
    ok, msg = _verify_admin_access(request, token=token, min_interval_sec=5.0)
    if not ok:
        return {"success": False, "error": msg}

    try:
        payload = await request.json()
        remote_sadang = payload.get("sadang_csv", "")
        if remote_sadang:
            merge_telemetry_csv("data/sadang_telemetry.csv", remote_sadang, lambda r: (r[0], r[5]), lambda r: r[0])

        remote_upstream = payload.get("upstream_csv", "")
        if remote_upstream:
            merge_telemetry_csv("data/upstream_telemetry.csv", remote_upstream, lambda r: (r[0], r[1], r[6]), lambda r: r[0])

        online_learner.rebuild_from_csvs()
        online_learner.save_state()
        CACHE_XAI_ANALYSIS.clear()
        return {
            "success": True,
            "message": "백업 데이터 병합 및 모델 재구축 완료",
            "summary": online_learner.get_status_summary()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/backup/export")
def export_backup_data(request: Request, token: str = Query("")):
    """서버에 축적된 텔레메트리 CSV 및 학습 모델 상태를 JSON으로 추출 (동기화 및 백업용)"""
    ok, msg = _verify_admin_access(request, token=token, min_interval_sec=2.0)
    if not ok:
        return {"success": False, "error": msg}

    sadang_csv = ""
    if os.path.exists("data/sadang_telemetry.csv"):
        try:
            with open("data/sadang_telemetry.csv", "r", encoding="utf-8-sig") as f:
                sadang_csv = f.read()
        except Exception:
            pass
    upstream_csv = ""
    if os.path.exists("data/upstream_telemetry.csv"):
        try:
            with open("data/upstream_telemetry.csv", "r", encoding="utf-8-sig") as f:
                upstream_csv = f.read()
        except Exception:
            pass
    model_json = {}
    if os.path.exists("models/sadang_online_model.json"):
        try:
            with open("models/sadang_online_model.json", "r", encoding="utf-8") as f:
                model_json = json.load(f)
        except Exception:
            pass
    return {
        "success": True,
        "sadang_csv": sadang_csv,
        "upstream_csv": upstream_csv,
        "model_json": model_json,
        "summary": online_learner.get_status_summary()
    }

@app.get("/api/timetable")
def get_timetable():
    """7790번 첫차/막차, 배차간격, MiRi 예약버스 고정 시간표"""
    return {
        "success": True,
        **TIMETABLE_INFO
    }

@app.get("/sw.js")
def get_service_worker():
    return Response(content="// no-op service worker", media_type="application/javascript")

if __name__ == "__main__":
    import uvicorn
    print("[INFO] 7790 타요 웹서버를 시작합니다: http://127.0.0.1:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
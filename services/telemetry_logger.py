import os
import csv
import time
import datetime
import threading
from typing import List, Dict, Any, Callable
from services.online_learner import online_learner, is_friday_night_window

TELEMETRY_LOCK = threading.Lock()
DATA_DIR = "data"
SADANG_CSV_PATH = os.path.join(DATA_DIR, "sadang_telemetry.csv")
UPSTREAM_CSV_PATH = os.path.join(DATA_DIR, "upstream_telemetry.csv")

SADANG_POST_DEPARTURE_STATION_IDS = {
    "277102049": "남현동(경유)",
    "277102291": "과천진입(경유)",
    "220000009": "과천동행정복지센터",
    "277103204": "대공원고가교(경유)",
}


SADANG_CHECKPOINT_OFFSETS_SEC = {
    "277102049": 60.0,   
    "277102291": 130.0,  
    "220000009": 210.0,  
    "277103204": 300.0,  
}


UPSTREAM_POST_STATION_MAP = {
    "233002371": ("233002245", "효행초등학교정문"),  # 아이파크 진입 시 -> 효행초 승차 완료 확정
    "233002255": ("233002371", "아이파크정문"),      # 신명아파트 진입 시 -> 아이파크 승차 완료 확정
    "233003021": ("233002255", "신명아파트"),        # 수원대입구 진입 시 -> 신명아파트 승차 완료 확정
    "233000945": ("233003021", "수원대입구"),        # 와우지구 진입 시 -> 수원대입구 승차 완료 확정
}

import json


TRIP_DEDUP_COOLDOWN_SEC = 2700.0
DEDUP_CACHE_PATH = os.path.join(DATA_DIR, "trip_dedup_cache.json")
_logged_sadang_trips: Dict[str, float] = {}     
_logged_upstream_trips: Dict[str, float] = {}   
_last_sadang_departure_time: float = 0.0
_dedup_loaded: bool = False


def _load_dedup_cache():
    global _logged_sadang_trips, _logged_upstream_trips, _last_sadang_departure_time, _dedup_loaded
    if _dedup_loaded:
        return
    _dedup_loaded = True
    try:
        if os.path.exists(DEDUP_CACHE_PATH):
            with open(DEDUP_CACHE_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
            now_epoch = time.time()
            _logged_sadang_trips = {
                k: float(v) for k, v in d.get("sadang", {}).items()
                if now_epoch - float(v) <= TRIP_DEDUP_COOLDOWN_SEC
            }
            _logged_upstream_trips = {
                k: float(v) for k, v in d.get("upstream", {}).items()
                if now_epoch - float(v) <= TRIP_DEDUP_COOLDOWN_SEC
            }
            _last_sadang_departure_time = float(d.get("last_sadang_departure_time", 0.0))
    except Exception:
        pass


def _save_dedup_cache():
    """원자적(Atomic) 파일 쓰기로 디스크 쿨다운 캐시 저장"""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = DEDUP_CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "sadang": _logged_sadang_trips,
                "upstream": _logged_upstream_trips,
                "last_sadang_departure_time": _last_sadang_departure_time
            }, f)
        os.replace(tmp, DEDUP_CACHE_PATH)
    except Exception:
        pass


def _ensure_csv_headers():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(SADANG_CSV_PATH):
        with open(SADANG_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "weekday", "is_friday_night", "hour", "minute",
                "plate", "is_double_decker", "total_seats", "remain_seats",
                "actual_boarded", "headway_gap_min", "heuristic_prior",
                "crowd_status", "checkpoint_station"
            ])
    if not os.path.exists(UPSTREAM_CSV_PATH):
        with open(UPSTREAM_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "station_id", "station_name", "weekday",
                "hour", "minute", "plate", "is_double_decker",
                "total_seats", "remain_seats", "consumed_seats"
            ])


def process_live_vehicles_for_learning(
    live_vehicles: List[Dict[str, Any]],
    now_dt: datetime.datetime,
    heuristic_prior_fn: Callable[[], tuple],
    crowd_status: str = "none"
):
    """
    GBIS 실시간 차량 목록을 스캔하여:
    1) 사당역 9번 출구를 막 출발한 하행(DOWN) 버스의 실제 탑승 인원을 역산(45/71 - 잔여석)하고,
    2) 상행(UP) 4개 주요 정류장 통과 버스의 누적 소진 좌석을 역산하여,
    CSV 데이터셋에 영구 기록함과 동시에 실시간 온라인 머신러닝 모델(online_learner)을 즉시 학습시킵니다.
    """
    global _last_sadang_departure_time
    if not live_vehicles:
        return

    now_epoch = time.time()
    _ensure_csv_headers()

    with TELEMETRY_LOCK:
        _load_dedup_cache()
        dirty_dedup = False
        expired_s = [k for k, ts in _logged_sadang_trips.items() if now_epoch - ts > TRIP_DEDUP_COOLDOWN_SEC]
        for k in expired_s:
            _logged_sadang_trips.pop(k, None)
            dirty_dedup = True

        expired_u = [k for k, ts in _logged_upstream_trips.items() if now_epoch - ts > TRIP_DEDUP_COOLDOWN_SEC]
        for k in expired_u:
            _logged_upstream_trips.pop(k, None)
            dirty_dedup = True

        for v in live_vehicles:
            if v.get("isReserved", False):
                continue

            plate = v.get("shortPlate") or v.get("plateNo") or ""
            if not plate:
                continue

            st_id = str(v.get("currentStationId", ""))
            is_dd = bool(v.get("isDoubleDecker", False))
            total_seats = 71 if is_dd else 45
            remain_seats = int(v.get("remainSeats", 0))

            # 비정상 센서값 방어
            if remain_seats < 0 or remain_seats > total_seats:
                continue


            if st_id in SADANG_POST_DEPARTURE_IDS_SET and plate not in _logged_sadang_trips:
                actual_boarded = max(0, total_seats - remain_seats)
                offset_sec = SADANG_CHECKPOINT_OFFSETS_SEC.get(st_id, 90.0)
                effective_dep_epoch = now_epoch - offset_sec

                if _last_sadang_departure_time > 0:
                    gap_min = (effective_dep_epoch - _last_sadang_departure_time) / 60.0
                    headway_gap_min = max(4.0, min(35.0, gap_min))
                else:
                    headway_gap_min = 12.0

                _last_sadang_departure_time = effective_dep_epoch
                _logged_sadang_trips[plate] = now_epoch
                dirty_dedup = True

                try:
                    prior_val, _ = heuristic_prior_fn()
                except Exception:
                    prior_val = 20

                t_float = now_dt.hour + now_dt.minute / 60.0
                fri_night = 1 if is_friday_night_window(now_dt.weekday(), t_float) else 0
                checkpoint_name = SADANG_POST_DEPARTURE_STATION_IDS.get(st_id, st_id)

                try:
                    with open(SADANG_CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            now_dt.strftime("%Y-%m-%d %H:%M:%S"),
                            now_dt.weekday(),
                            fri_night,
                            now_dt.hour,
                            now_dt.minute,
                            plate,
                            1 if is_dd else 0,
                            total_seats,
                            remain_seats,
                            actual_boarded,
                            round(headway_gap_min, 1),
                            prior_val,
                            crowd_status,
                            checkpoint_name
                        ])
                except Exception as e_csv:
                    print(f"[WARN] 사당역 텔레메트리 CSV 기록 오류: {e_csv}")

                try:
                    online_learner.learn_from_sadang_bus(
                        now_dt=now_dt,
                        plate=plate,
                        actual_boarded=actual_boarded,
                        remain_seats=remain_seats,
                        is_double_decker=is_dd,
                        headway_gap_min=headway_gap_min,
                        heuristic_prior=prior_val,
                        crowd_status=crowd_status
                    )
                    print(
                        f"[ONLINE-ML] 사당역 출발 버스({plate}) 실시간 학습 완료! "
                        f"실제 탑승: {actual_boarded}명 (잔여 {remain_seats}/{total_seats}석, 배차간격 {headway_gap_min:.1f}분)"
                    )
                except Exception as e_ml:
                    print(f"[WARN] 온라인 학습 업데이트 오류: {e_ml}")

            elif st_id in UPSTREAM_POST_STATION_MAP:
                target_st_id, target_st_name = UPSTREAM_POST_STATION_MAP[st_id]
                dedup_key = f"{target_st_id}_{plate}"
                if dedup_key not in _logged_upstream_trips:
                    _logged_upstream_trips[dedup_key] = now_epoch
                    dirty_dedup = True
                    consumed_seats = max(0, total_seats - remain_seats)

                    try:
                        with open(UPSTREAM_CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                now_dt.strftime("%Y-%m-%d %H:%M:%S"),
                                target_st_id,
                                target_st_name,
                                now_dt.weekday(),
                                now_dt.hour,
                                now_dt.minute,
                                plate,
                                1 if is_dd else 0,
                                total_seats,
                                remain_seats,
                                consumed_seats
                            ])
                    except Exception:
                        pass

                    try:
                        online_learner.learn_from_upstream_bus(
                            station_id=target_st_id,
                            now_dt=now_dt,
                            plate=plate,
                            consumed_seats=consumed_seats,
                            remain_seats=remain_seats,
                            is_double_decker=is_dd
                        )
                    except Exception:
                        pass

        if dirty_dedup:
            _save_dedup_cache()


def merge_telemetry_csv(file_path: str, remote_text: str, key_fn: Callable, sort_fn: Callable) -> int:
    """원격/백업 CSV 텍스트와 로컬 CSV 파일을 중복 없이 병합 후 원자적(Atomic)으로 저장"""
    import io
    rows = []
    seen = set()
    header = None
    with TELEMETRY_LOCK:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8-sig") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for r in reader:
                        if not r:
                            continue
                        try:
                            k = key_fn(r)
                            if k not in seen:
                                seen.add(k)
                                rows.append(r)
                        except Exception:
                            pass
            except Exception:
                pass

        remote_reader = csv.reader(io.StringIO(remote_text))
        remote_header = next(remote_reader, None)
        if not header:
            header = remote_header
        for r in remote_reader:
            if not r:
                continue
            try:
                k = key_fn(r)
                if k not in seen:
                    seen.add(k)
                    rows.append(r)
            except Exception:
                pass

        rows.sort(key=sort_fn)
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        tmp_path = file_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(header)
            writer.writerows(rows)
        os.replace(tmp_path, file_path)
    return len(rows)


SADANG_POST_DEPARTURE_IDS_SET = set(SADANG_POST_DEPARTURE_STATION_IDS.keys())

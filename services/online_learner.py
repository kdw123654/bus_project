import os
import json
import math
import time
import threading
import datetime
from typing import Optional, Dict, Any, Tuple, List, Set

try:
    import numpy as np
    from sklearn.linear_model import Ridge
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

MODEL_LOCK = threading.Lock()
DATA_DIR = "data"
MODELS_DIR = "models"
MODEL_STATE_FILE = os.path.join(MODELS_DIR, "sadang_online_model.json")

RBF_CENTERS = [
    6.25,   # 06:15 첫차/이른 통학
    7.65,   # 07:40 1교시 등교 급증
    8.25,   # 08:15 1교시 등교 최대 극피크
    9.15,   # 09:10 2교시 등교 피크
    10.50,  # 10:30 오전 분산 통학
    13.00,  # 13:00 점심/오후 시간대
    15.50,  # 15:30 오후 한산 시간대
    17.00,  # 17:00 퇴근 유입 시작
    18.25,  # 18:15 퇴근길 최대 극피크
    19.25,  # 19:15 퇴근길 잔여 피크
    21.75,  # 21:45 야간 / 금요일 1차 모임 귀가 급증
    23.00,  # 23:00 심야 / 금요일 2차 귀가 극피크
    24.15,  # 00:10 사당역 00:30 막차 대란
]


def is_friday_night_window(weekday: int, time_float: float) -> bool:
    """금요일 심야(21:30~00:40) 시간대 여부 판별"""
    if weekday == 4 and time_float >= 21.5:
        return True
    if weekday == 5 and time_float <= 0.67:
        return True
    return False


def get_effective_day_and_time(weekday: int, hour: int, minute: int) -> Tuple[int, float]:
    """자정 이후(00:00~02:00) 심야 운행을 전일 24~26시로 보정"""
    t_float = hour + (minute / 60.0)
    if t_float < 2.0:
        eff_weekday = (weekday - 1) % 7
        eff_time = t_float + 24.0
        return eff_weekday, eff_time
    return weekday, t_float


def get_day_group(eff_weekday: int, eff_time: float) -> str:
    """
    3분할 요일군 판별:
    - 'weekday': 월~목 (정규 등하교 통학 및 일반 출퇴근)
    - 'friday': 금요일 (오후 본가 귀향 및 심야 21:30~00:40 회식/모임 피크)
    - 'weekend': 토·일 (주말 비통학 감차 운행 및 일반 나들이)
    """
    if eff_weekday in [0, 1, 2, 3]:
        return "weekday"
    elif eff_weekday == 4:
        return "friday"
    else:
        return "weekend"


class DayGroupState:
    """요일군별 독립 온라인 학습 상태 (가중치, 통계 버킷, 잔차 추적)"""
    def __init__(self, name: str, display_name: str, num_features: int = 24):
        self.name = name
        self.display_name = display_name
        self.num_features = num_features
        self.weights = [0.0] * num_features
        self.bias = 0.0
        self.learning_rate = 0.015
        self.ewma_residual = 0.0
        self.last_departure_epoch = 0.0
        self.learned_buses = 0
        self.cumulative_abs_error = 0.0
        self.within_3_count = 0
        self.eval_count = 0
        self.history_X: List[List[float]] = []
        self.history_y: List[float] = []
        self.sklearn_ridge = None
        self.bucket_stats: Dict[str, Dict[str, float]] = {}
        self.last_bus_event: Optional[Dict[str, Any]] = None

    def dot_predict_delta(self, feats: List[float]) -> float:
        if self.sklearn_ridge is not None and HAS_SKLEARN and len(self.history_y) >= 15:
            try:
                pred = float(self.sklearn_ridge.predict([feats])[0])
                return max(-25.0, min(30.0, pred))
            except Exception:
                pass
        val = self.bias + sum(w * x for w, x in zip(self.weights, feats))
        return max(-25.0, min(30.0, val))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "weights": [round(w, 6) for w in self.weights],
            "bias": round(self.bias, 6),
            "ewma_residual": round(self.ewma_residual, 4),
            "last_departure_epoch": self.last_departure_epoch,
            "learned_buses": self.learned_buses,
            "cumulative_abs_error": round(self.cumulative_abs_error, 4),
            "within_3_count": self.within_3_count,
            "eval_count": self.eval_count,
            "bucket_stats": self.bucket_stats,
            "last_bus_event": self.last_bus_event
        }

    def from_dict(self, d: Dict[str, Any]):
        loaded_w = d.get("weights", [])
        if len(loaded_w) == self.num_features:
            self.weights = [float(x) for x in loaded_w]
        self.bias = float(d.get("bias", 0.0))
        self.ewma_residual = float(d.get("ewma_residual", 0.0))
        self.last_departure_epoch = float(d.get("last_departure_epoch", 0.0))
        self.learned_buses = int(d.get("learned_buses", 0))
        self.cumulative_abs_error = float(d.get("cumulative_abs_error", 0.0))
        self.within_3_count = int(d.get("within_3_count", 0))
        self.eval_count = int(d.get("eval_count", 0))
        self.bucket_stats = d.get("bucket_stats", {})
        self.last_bus_event = d.get("last_bus_event", None)


class SadangOnlineLearner:
    """
    [7790 실시간 온라인 머신러닝 학습 엔진 - 요일군 3분할 버전]
    - 평일(월~목), 금요일, 주말(토·일)의 데이터 간섭을 원천 차단하고 독립 모델로 분리 학습합니다.
    - Cold-start 방지: 평일 데이터가 0건일 때 도메인 경험 기본값(Prior)을 100% 신뢰하여 오차를 방지합니다.
    """
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
        self.num_features = 24
        self.groups: Dict[str, DayGroupState] = {
            "weekday": DayGroupState("weekday", "평일(월~목)", self.num_features),
            "friday": DayGroupState("friday", "금요일", self.num_features),
            "weekend": DayGroupState("weekend", "주말(토·일)", self.num_features)
        }
        self.upstream_bucket_stats: Dict[str, Dict[str, float]] = {}
        self.group_upstream_learned: Dict[str, int] = {
            "weekday": 0,
            "friday": 0,
            "weekend": 0
        }
        self.total_learned_buses: int = 0
        self.total_upstream_learned: int = 0
        self.upstream_learned_trips: Set[str] = set()
        self.last_bus_event: Optional[Dict[str, Any]] = None
        self.last_upstream_bus_event: Optional[Dict[str, Any]] = None
        self.last_updated_str: str = "초기화 대기 중"

        self.load_state()

    def _get_current_group(self, now_dt: Optional[datetime.datetime] = None) -> DayGroupState:
        if now_dt is None:
            now_dt = datetime.datetime.now()
        eff_weekday, eff_time = get_effective_day_and_time(now_dt.weekday(), now_dt.hour, now_dt.minute)
        group_key = get_day_group(eff_weekday, eff_time)
        return self.groups[group_key]

    @property
    def bucket_stats(self) -> Dict[str, Dict[str, float]]:
        merged = {}
        for g in self.groups.values():
            merged.update(g.bucket_stats)
        return merged

    @property
    def weights(self) -> List[float]:
        return self._get_current_group().weights

    @property
    def bias(self) -> float:
        return self._get_current_group().bias

    @property
    def ewma_residual(self) -> float:
        return self._get_current_group().ewma_residual

    @property
    def last_departure_epoch(self) -> float:
        return self._get_current_group().last_departure_epoch

    @property
    def cumulative_abs_error(self) -> float:
        return sum(g.cumulative_abs_error for g in self.groups.values())

    @property
    def eval_count(self) -> int:
        return sum(g.eval_count for g in self.groups.values())

    @property
    def within_3_count(self) -> int:
        return sum(g.within_3_count for g in self.groups.values())

    def extract_features(
        self,
        eff_weekday: int,
        eff_time: float,
        headway_gap_min: float = 12.0,
        is_double_decker: bool = False
    ) -> List[float]:
        """24차원 특징 벡터 추출 (RBF 시간 기저 + 삼각함수 + 요일 원핫 + 배차간격 + 차종)"""
        rbf_feats = []
        for c in RBF_CENTERS:
            dist = eff_time - c
            rbf_feats.append(math.exp(-(dist * dist) / (2.0 * (0.75 ** 2))))

        rad = 2.0 * math.pi * (eff_time / 24.0)
        sin_t = math.sin(rad)
        cos_t = math.cos(rad)
        is_mon_tue = 1.0 if eff_weekday in [0, 1] else 0.0
        is_wed = 1.0 if eff_weekday == 2 else 0.0
        is_thu = 1.0 if eff_weekday == 3 else 0.0
        is_fri = 1.0 if eff_weekday == 4 else 0.0
        is_sat = 1.0 if eff_weekday == 5 else 0.0
        is_sun = 1.0 if eff_weekday == 6 else 0.0

        is_fri_night = 1.0 if (eff_weekday == 4 and eff_time >= 21.5) else 0.0
        headway_norm = max(0.4, min(2.5, headway_gap_min / 12.0)) - 1.0
        dd_feat = 1.0 if is_double_decker else 0.0

        feats = (
            rbf_feats
            + [sin_t, cos_t, is_mon_tue, is_wed, is_thu, is_fri, is_sat, is_sun, is_fri_night, headway_norm, dd_feat]
        )
        return feats[:self.num_features]

    def learn_from_sadang_bus(
        self,
        now_dt: datetime.datetime,
        plate: str,
        actual_boarded: int,
        remain_seats: int,
        is_double_decker: bool,
        headway_gap_min: float,
        heuristic_prior: int,
        crowd_status: str = "none"
    ) -> Dict[str, Any]:
        """사당역 통과 버스 실측 데이터를 해당 요일군 모델에 독립 학습"""
        with MODEL_LOCK:
            weekday = now_dt.weekday()
            eff_weekday, eff_time = get_effective_day_and_time(weekday, now_dt.hour, now_dt.minute)
            group_key = get_day_group(eff_weekday, eff_time)
            group = self.groups[group_key]
            slot_idx = int(eff_time * 2)
            slot_key = str(slot_idx)

            effective_demand = float(actual_boarded)
            if remain_seats == 0:
                censor_bonus = 8.0 if crowd_status == "crowded" else 5.0
                effective_demand += censor_bonus

            current_pred, _, _ = self._predict_internal(
                now_dt=now_dt,
                heuristic_base=heuristic_prior,
                headway_gap_min=headway_gap_min,
                is_double_decker=is_double_decker
            )
            abs_err = abs(current_pred - actual_boarded)
            group.eval_count += 1
            group.cumulative_abs_error += abs_err
            if abs_err <= 4:
                group.within_3_count += 1

            b_rec = group.bucket_stats.get(slot_key, {"count": 0, "mean": float(heuristic_prior), "ewma": float(heuristic_prior)})
            c = b_rec["count"] + 1
            new_mean = b_rec["mean"] + (effective_demand - b_rec["mean"]) / c
            ewma_alpha = 0.35 if c > 1 else 1.0
            new_ewma = (1.0 - ewma_alpha) * b_rec["ewma"] + ewma_alpha * effective_demand
            group.bucket_stats[slot_key] = {
                "count": c,
                "mean": round(new_mean, 2),
                "ewma": round(new_ewma, 2),
                "last_actual": actual_boarded
            }

            target_delta = effective_demand - float(heuristic_prior)
            feats = self.extract_features(eff_weekday, eff_time, headway_gap_min, is_double_decker)
            pred_delta = sum(w * x for w, x in zip(group.weights, feats)) + group.bias
            err = target_delta - pred_delta

            lr = group.learning_rate / (1.0 + 0.002 * group.learned_buses)
            for i in range(self.num_features):
                grad = -err * feats[i] + 0.01 * group.weights[i]
                group.weights[i] = max(-15.0, min(15.0, group.weights[i] - lr * grad))
            group.bias = max(-15.0, min(15.0, group.bias + lr * err * 0.5))

            group.history_X.append(feats)
            group.history_y.append(target_delta)
            if len(group.history_y) > 2000:
                group.history_X = group.history_X[-2000:]
                group.history_y = group.history_y[-2000:]
            if HAS_SKLEARN and len(group.history_y) >= 15:
                try:
                    ridge = Ridge(alpha=2.0)
                    ridge.fit(np.array(group.history_X), np.array(group.history_y))
                    group.sklearn_ridge = ridge
                except Exception:
                    pass

            now_epoch = time.time()
            instant_residual = effective_demand - float(heuristic_prior)
            if group.last_departure_epoch > 0 and (now_epoch - group.last_departure_epoch < 3600):
                group.ewma_residual = 0.45 * group.ewma_residual + 0.55 * instant_residual
            else:
                group.ewma_residual = instant_residual

            group.last_departure_epoch = now_epoch
            group.learned_buses += 1
            self.total_learned_buses = sum(g.learned_buses for g in self.groups.values())
            self.last_updated_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

            event_data = {
                "plate": plate,
                "actualBoarded": actual_boarded,
                "remainSeats": remain_seats,
                "isDoubleDecker": is_double_decker,
                "headwayGapMin": round(headway_gap_min, 1),
                "timeStr": now_dt.strftime("%H:%M"),
                "weekday": eff_weekday,
                "dayGroup": group_key,
                "timestamp": now_epoch
            }
            group.last_bus_event = event_data
            self.last_bus_event = event_data

            self.save_state()
            return self.get_status_summary()

    def learn_from_upstream_bus(
        self,
        station_id: str,
        now_dt: datetime.datetime,
        plate: str,
        consumed_seats: int,
        remain_seats: int,
        is_double_decker: bool
    ):
        """수원대·봉담 상류 4개 정류장 통과 버스를 해당 요일군 모델에 독립 학습"""
        with MODEL_LOCK:
            eff_weekday, eff_time = get_effective_day_and_time(now_dt.weekday(), now_dt.hour, now_dt.minute)
            group_key = get_day_group(eff_weekday, eff_time)
            slot_idx = int(eff_time * 2)
            key = f"{station_id}_{group_key}_{slot_idx}"

            rec = self.upstream_bucket_stats.get(key, {"count": 0, "ewma_consumed": float(consumed_seats)})
            c = rec["count"] + 1
            alpha = 0.35 if c > 1 else 1.0
            new_ewma = (1.0 - alpha) * rec["ewma_consumed"] + alpha * float(consumed_seats)
            self.upstream_bucket_stats[key] = {
                "count": c,
                "ewma_consumed": round(new_ewma, 2),
                "last_remain": remain_seats,
                "last_plate": plate,
                "updated_at": now_dt.strftime("%H:%M"),
                "dayGroup": group_key
            }

            trip_key = f"{plate}_{now_dt.strftime('%Y%m%d%H')}"
            if trip_key not in self.upstream_learned_trips:
                self.upstream_learned_trips.add(trip_key)
                self.total_upstream_learned += 1
                self.group_upstream_learned[group_key] = self.group_upstream_learned.get(group_key, 0) + 1

            st_names = {
                "233002245": "효행초",
                "233002371": "아이파크",
                "233002255": "신명아파트",
                "233003021": "수원대입구"
            }
            self.last_upstream_bus_event = {
                "stationId": station_id,
                "stationName": st_names.get(station_id, "수원대·봉담"),
                "plate": plate,
                "consumed": consumed_seats,
                "remain": remain_seats,
                "isDoubleDecker": is_double_decker,
                "timeStr": now_dt.strftime("%H:%M"),
                "dayGroup": group_key
            }
            self.save_state()

    def _predict_internal(
        self,
        now_dt: datetime.datetime,
        heuristic_base: int,
        headway_gap_min: Optional[float] = None,
        is_double_decker: bool = False
    ) -> Tuple[int, str, Dict[str, Any]]:
        """내부 잔차 예측 (해당 요일군 모델만 활용, 데이터 부족 시 기본값 100% 보존)"""
        weekday = now_dt.weekday()
        eff_weekday, eff_time = get_effective_day_and_time(weekday, now_dt.hour, now_dt.minute)
        group_key = get_day_group(eff_weekday, eff_time)
        group = self.groups[group_key]
        slot_idx = int(eff_time * 2)
        slot_key = str(slot_idx)

        now_epoch = time.time()
        if headway_gap_min is None:
            if group.last_departure_epoch > 0:
                elapsed_min = (now_epoch - group.last_departure_epoch) / 60.0
                headway_gap_min = max(6.0, min(28.0, elapsed_min))
            else:
                headway_gap_min = 12.0

        b_rec = group.bucket_stats.get(slot_key)
        bucket_count = b_rec["count"] if b_rec else 0
        bucket_val = b_rec["ewma"] if b_rec else float(heuristic_base)
        feats = self.extract_features(eff_weekday, eff_time, headway_gap_min, is_double_decker)
        reg_delta = group.dot_predict_delta(feats)
        reg_val = float(heuristic_base) + reg_delta

        global_alpha = min(0.80, group.learned_buses / 60.0) if group.learned_buses > 0 else 0.0
        bucket_alpha = min(0.85, bucket_count / 5.0) if bucket_count > 0 else 0.0

        if bucket_count > 0:
            ml_combined = 0.65 * bucket_val + 0.35 * reg_val
            effective_alpha = max(global_alpha * 0.6, bucket_alpha)
        else:
            ml_combined = reg_val
            effective_alpha = global_alpha * 0.5

        # 데이터가 없을 때는 100% 경험 기본값 유지 (Cold-start 방어)
        blended = (1.0 - effective_alpha) * float(heuristic_base) + effective_alpha * ml_combined

        momentum_shift = 0.0
        recent_bus_note = ""
        if group.last_departure_epoch > 0:
            elapsed_min = (now_epoch - group.last_departure_epoch) / 60.0
            if 0 <= elapsed_min <= 35.0:
                decay = math.exp(-elapsed_min / 20.0)
                residual_damping = max(0.15, 1.0 - effective_alpha)
                momentum_shift = max(-10.0, min(14.0, group.ewma_residual * 0.45 * decay * residual_damping))
                if group.last_bus_event:
                    last_p = group.last_bus_event.get("plate", "")
                    last_b = group.last_bus_event.get("actualBoarded", 0)
                    recent_bus_note = f" · 직전({last_p}) {last_b}명 실측 반영"

        final_est = int(round(max(2.0, min(68.0, blended + momentum_shift))))

        ml_meta = {
            "totalLearnedBuses": self.total_learned_buses,
            "groupLearnedBuses": group.learned_buses,
            "dayGroup": group_key,
            "bucketLearnedCount": bucket_count,
            "mlBlendWeight": round(effective_alpha * 100, 1),
            "momentumShift": round(momentum_shift, 1),
            "lastBusEvent": group.last_bus_event,
            "recentBusNote": recent_bus_note
        }
        return final_est, recent_bus_note, ml_meta

    def predict_sadang_waiters(
        self,
        now_dt: datetime.datetime,
        heuristic_base: int,
        heuristic_desc: str,
        headway_gap_min: Optional[float] = None,
        is_double_decker: bool = False
    ) -> Tuple[int, str, Dict[str, Any]]:
        """사당역 대기 승객 수 예측 (요일군 분리 적용)"""
        with MODEL_LOCK:
            final_est, recent_note, ml_meta = self._predict_internal(
                now_dt=now_dt,
                heuristic_base=heuristic_base,
                headway_gap_min=headway_gap_min,
                is_double_decker=is_double_decker
            )
            group_key = ml_meta.get("dayGroup", "weekday")
            group_learned = ml_meta.get("groupLearnedBuses", 0)
            group_label = {"weekday": "평일", "friday": "금요일", "weekend": "주말"}.get(group_key, "")

            if group_learned > 0:
                enriched_desc = f"{heuristic_desc} [{group_label} 온라인ML {group_learned}대 학습{recent_note}]"
            else:
                enriched_desc = heuristic_desc
            return final_est, enriched_desc, ml_meta

    def get_upstream_learned_consumption(
        self,
        station_id: str,
        now_dt: datetime.datetime,
        default_consumed: int
    ) -> Tuple[int, Optional[str]]:
        """상류 4개 정류장 소진 좌석 수 조회 (요일군 분리 적용)"""
        with MODEL_LOCK:
            eff_weekday, eff_time = get_effective_day_and_time(now_dt.weekday(), now_dt.hour, now_dt.minute)
            group_key = get_day_group(eff_weekday, eff_time)
            slot_idx = int(eff_time * 2)
            key = f"{station_id}_{group_key}_{slot_idx}"
            rec = self.upstream_bucket_stats.get(key)
            if not rec or rec.get("count", 0) == 0:
                legacy_key = f"{station_id}_{eff_weekday}_{slot_idx}"
                rec = self.upstream_bucket_stats.get(legacy_key)
                if not rec or rec.get("count", 0) == 0:
                    return default_consumed, None

            cnt = rec["count"]
            alpha = min(0.75, cnt / 5.0)
            blended = int(round((1.0 - alpha) * default_consumed + alpha * rec["ewma_consumed"]))
            group_label = {"weekday": "평일", "friday": "금요일", "weekend": "주말"}.get(group_key, "")
            note = f"실시간 온라인 학습({group_label} {cnt}회 실측 평균 소진 {int(round(rec['ewma_consumed']))}석) 융합"
            return blended, note

    def get_status_summary(self) -> Dict[str, Any]:
        """API 및 대시보드 상태 요약 (기존 인터페이스 및 3분할 세부 정보 완전 호환)"""
        curr_group = self._get_current_group()
        total_eval = sum(g.eval_count for g in self.groups.values())
        total_err = sum(g.cumulative_abs_error for g in self.groups.values())
        total_within_3 = sum(g.within_3_count for g in self.groups.values())

        mae = round(total_err / total_eval, 2) if total_eval > 0 else None
        hit_rate = round((total_within_3 / total_eval) * 100, 1) if total_eval > 0 else None

        active_alpha_pct = int(round(max(15.0, min(80.0, (curr_group.learned_buses / 60.0) * 100.0)))) if curr_group.learned_buses > 0 else 0
        overall_alpha_pct = int(round(max(15.0, min(80.0, (self.total_learned_buses / 80.0) * 100.0)))) if self.total_learned_buses > 0 else 0

        last_rec = None
        if self.last_bus_event:
            last_rec = {
                "plate": self.last_bus_event.get("plate", ""),
                "boarded": self.last_bus_event.get("actualBoarded", 0),
                "remain": self.last_bus_event.get("remainSeats", 0),
                "timeStr": self.last_bus_event.get("timeStr", ""),
                "dayGroup": self.last_bus_event.get("dayGroup", "")
            }
        last_up_rec = None
        if self.last_upstream_bus_event:
            last_up_rec = {
                "stationId": self.last_upstream_bus_event.get("stationId", ""),
                "stationName": self.last_upstream_bus_event.get("stationName", "수원대·봉담"),
                "plate": self.last_upstream_bus_event.get("plate", ""),
                "consumed": self.last_upstream_bus_event.get("consumed", 0),
                "remain": self.last_upstream_bus_event.get("remain", 0),
                "timeStr": self.last_upstream_bus_event.get("timeStr", ""),
                "dayGroup": self.last_upstream_bus_event.get("dayGroup", "")
            }

        day_groups_summary = {}
        for k, g in self.groups.items():
            g_alpha = int(round(max(15.0, min(80.0, (g.learned_buses / 60.0) * 100.0)))) if g.learned_buses > 0 else 0
            g_mae = round(g.cumulative_abs_error / g.eval_count, 2) if g.eval_count > 0 else None
            day_groups_summary[k] = {
                "name": g.display_name,
                "sadangLearned": g.learned_buses,
                "upstreamLearned": self.group_upstream_learned.get(k, 0),
                "activeBuckets": len(g.bucket_stats),
                "mlWeightPct": g_alpha,
                "mae": g_mae
            }

        return {
            "mode": "요일별 실시간 온라인 AI 학습 (Online SGD + Empirical Bayes)",
            "engine": "Online Incremental SGD + Empirical Bayes Hybrid",
            "coverage": "월~목 통학·출퇴근 / 금요일 귀향·심야피크 / 토·일 주말감차 독립 학습",
            "currentDayGroup": curr_group.name,
            "currentDayGroupName": curr_group.display_name,
            "totalSadangLearned": self.total_learned_buses,
            "totalUpstreamLearned": self.total_upstream_learned,
            "totalLearnedAll": self.total_learned_buses + self.total_upstream_learned,
            "sadangTotalLearnedBuses": self.total_learned_buses,
            "upstreamTotalLearnedBuses": self.total_upstream_learned,
            "mlWeightPct": active_alpha_pct,
            "overallMlWeightPct": overall_alpha_pct,
            "activeTimeBuckets": len(curr_group.bucket_stats),
            "maeSeats": mae,
            "within4SeatsAccuracy": hit_rate,
            "lastUpdated": self.last_updated_str,
            "dayGroups": day_groups_summary,
            "lastBusEvent": self.last_bus_event,
            "lastSadangRecord": last_rec,
            "lastUpstreamBusEvent": self.last_upstream_bus_event,
            "lastUpstreamRecord": last_up_rec
        }

    def save_state(self):
        """3분할 요일군 모델 및 통계 디스크 영구 저장"""
        try:
            state = {
                "day_groups": {k: g.to_dict() for k, g in self.groups.items()},
                "upstream_bucket_stats": self.upstream_bucket_stats,
                "group_upstream_learned": self.group_upstream_learned,
                "total_learned_buses": self.total_learned_buses,
                "total_upstream_learned": self.total_upstream_learned,
                "upstream_learned_trips": list(self.upstream_learned_trips)[-500:],
                "last_updated_str": self.last_updated_str,
                "last_bus_event": self.last_bus_event,
                "last_upstream_bus_event": self.last_upstream_bus_event
            }
            tmp_file = MODEL_STATE_FILE + ".tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            os.replace(tmp_file, MODEL_STATE_FILE)
        except Exception as e:
            print(f"[WARN] 온라인 학습 모델 저장 오류: {e}")

    def load_state(self):
        """서버 재시작 시 3분할 모델 복원 (구버전 포맷 감지 시 CSV 기반 원자적 재구축)"""
        try:
            if not os.path.exists(MODEL_STATE_FILE):
                self.rebuild_from_csvs()
                return

            with open(MODEL_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)

            if "day_groups" in state:
                for k, g_data in state["day_groups"].items():
                    if k in self.groups:
                        self.groups[k].from_dict(g_data)
                self.upstream_bucket_stats = state.get("upstream_bucket_stats", {})
                self.group_upstream_learned = state.get("group_upstream_learned", {"weekday": 0, "friday": 0, "weekend": 0})
                self.total_learned_buses = sum(g.learned_buses for g in self.groups.values())
                self.total_upstream_learned = int(state.get("total_upstream_learned", 0))
                self.upstream_learned_trips = set(state.get("upstream_learned_trips", []))
                self.last_updated_str = state.get("last_updated_str", "복원 완료")
                self.last_bus_event = state.get("last_bus_event", None)
                self.last_upstream_bus_event = state.get("last_upstream_bus_event", None)
            else:
                # 구버전 단일 모델 포맷 -> CSV 텔레메트리 기반으로 3분할 재구축
                self.rebuild_from_csvs()

            print(
                f"[OK] 요일군 3분할 온라인 학습 모델 복원 완료! "
                f"(사당역 {self.total_learned_buses}대 / 상행 {self.total_upstream_learned}대)"
            )
        except Exception as e:
            print(f"[WARN] 온라인 학습 모델 복원 오류: {e}")
            self.rebuild_from_csvs()

    def rebuild_from_csvs(self):
        """기존 텔레메트리 CSV 파일들을 읽어 3분할 요일군 모델에 정확히 분배 학습"""
        self.groups = {
            "weekday": DayGroupState("weekday", "평일(월~목)", self.num_features),
            "friday": DayGroupState("friday", "금요일", self.num_features),
            "weekend": DayGroupState("weekend", "주말(토·일)", self.num_features)
        }
        self.upstream_bucket_stats = {}
        self.group_upstream_learned = {"weekday": 0, "friday": 0, "weekend": 0}
        self.total_learned_buses = 0
        self.total_upstream_learned = 0
        self.upstream_learned_trips = set()

        sadang_csv = os.path.join(DATA_DIR, "sadang_telemetry.csv")
        up_csv = os.path.join(DATA_DIR, "upstream_telemetry.csv")

        # 1. 사당역 텔레메트리 분배 학습
        if os.path.exists(sadang_csv):
            try:
                import csv
                with open(sadang_csv, "r", encoding="utf-8-sig") as f:
                    rows = list(csv.DictReader(f))
                for r in rows:
                    try:
                        ts = datetime.datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")
                        plate = r["plate"]
                        actual = int(r["actual_boarded"])
                        remain = int(r["remain_seats"])
                        dd = bool(int(r.get("is_double_decker", "0")))
                        headway = float(r.get("headway_gap_min", "12.0"))
                        prior = int(r.get("heuristic_prior", "14"))
                        crowd = r.get("crowd_status", "none")
                        self.learn_from_sadang_bus(
                            now_dt=ts,
                            plate=plate,
                            actual_boarded=actual,
                            remain_seats=remain,
                            is_double_decker=dd,
                            headway_gap_min=headway,
                            heuristic_prior=prior,
                            crowd_status=crowd
                        )
                    except Exception:
                        pass
            except Exception as e:
                print(f"[WARN] 사당 텔레메트리 복원 실패: {e}")

        # 2. 상행 텔레메트리 분배 학습
        if os.path.exists(up_csv):
            try:
                import csv
                with open(up_csv, "r", encoding="utf-8-sig") as f:
                    rows = list(csv.DictReader(f))
                for r in rows:
                    try:
                        ts = datetime.datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")
                        st_id = r["station_id"]
                        plate = r["plate"]
                        consumed = int(r["consumed_seats"])
                        remain = int(r["remain_seats"])
                        dd = bool(int(r.get("is_double_decker", "0")))
                        self.learn_from_upstream_bus(
                            station_id=st_id,
                            now_dt=ts,
                            plate=plate,
                            consumed_seats=consumed,
                            remain_seats=remain,
                            is_double_decker=dd
                        )
                    except Exception:
                        pass
            except Exception as e:
                print(f"[WARN] 상행 텔레메트리 복원 실패: {e}")

        self.save_state()


online_learner = SadangOnlineLearner()

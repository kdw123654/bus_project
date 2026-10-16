import os
import json
import math
import time
import threading
import datetime
from typing import Optional, Dict, Any, Tuple, List

# 선택적 고성능 ML 라이브러리 로드 (미설치 환경에서도 순수 파이썬 온라인 SGD/Ridge 엔진으로 100% 무중단 동작)
try:
    import numpy as np
    from sklearn.linear_model import SGDRegressor, Ridge
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

MODEL_LOCK = threading.Lock()
DATA_DIR = "data"
MODELS_DIR = "models"
MODEL_STATE_FILE = os.path.join(MODELS_DIR, "sadang_online_model.json")

# 전 시간대(05:00 ~ 24:30) 비선형 피크 곡선을 정밀 근사하기 위한 13개 가우시안 RBF 중심 시간(시)
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
    21.75,  # 21:45 야간 / 불금 1차 모임 귀가 급증
    23.00,  # 23:00 심야 / 불금 2차 귀가 극피크
    24.15,  # 00:10 사당역 00:30 막차 대란
]


def is_friday_night_window(weekday: int, time_float: float) -> bool:
    """
    금요일 밤 21:30 ~ 토요일 새벽 00:40(금요일 자정 넘긴 막차 시간대)을
    '불금 심야 피크(Friday Night Peak)'로 정밀 판별합니다.
    """
    if weekday == 4 and time_float >= 21.5:
        return True
    if weekday == 5 and time_float <= 0.67:  # 금요일 밤이 자정을 넘어 토요일 00:00~00:40이 된 경우
        return True
    return False


def get_effective_day_and_time(weekday: int, hour: int, minute: int) -> Tuple[int, float]:
    """
    자정 직후(00:00 ~ 01:30) 심야 막차 운행을 전날 밤의 연장선(24.0 ~ 25.5시)으로 정규화합니다.
    예: 토요일 00:20 -> 금요일(weekday=4)의 24.33시로 매핑하여 불금 막차 연속성 보장.
    """
    t_float = hour + (minute / 60.0)
    if t_float < 2.0:
        eff_weekday = (weekday - 1) % 7
        eff_time = t_float + 24.0
        return eff_weekday, eff_time
    return weekday, t_float


class SadangOnlineLearner:
    """
    [전 요일 · 전 시간대 실시간 온라인 머신러닝 학습기 (Online / Incremental Learner)]
    - 버스가 사당역 및 주요 정류장을 통과할 때마다 실시간으로 실제 탑승 인원을 역산하여 즉시 학습합니다.
    - 1) 요일×30분 단위 Empirical Bayes 사후 분포 테이블 (Contextual Time-Slot Memory)
    - 2) 24차원 시계열/요일/배차간격/불금 피처 기반 온라인 SGD & Ridge 회귀 모델
    - 3) 직전 통과 차량의 실측 오차(Momentum Residual) 실시간 피드백
    """

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)

        # 1. 요일별·30분 시간대별 누적 학습 버킷: key = f"{eff_weekday}_{slot_idx}"
        self.bucket_stats: Dict[str, Dict[str, float]] = {}

        # 2. 상행 정류장(효행초/아이파크/신명/수원대) 실시간 보정 버킷: key = f"{station_id}_{eff_weekday}_{slot_idx}"
        self.upstream_bucket_stats: Dict[str, Dict[str, float]] = {}

        # 3. 온라인 선형/비선형 커널 회귀 가중치 (24차원 피처)
        self.num_features = 24
        self.weights: List[float] = [0.0] * self.num_features
        self.bias: float = 0.0
        self.learning_rate: float = 0.015

        # 4. 실시간 모멘텀 (가장 최근 사당역을 출발한 버스의 실측 탑승 정보 및 예측 오차)
        self.last_bus_event: Optional[Dict[str, Any]] = None
        self.ewma_residual: float = 0.0
        self.last_departure_epoch: float = 0.0

        # 5. 모델 메타데이터 및 정확도 추적 (MAE, 적중률)
        self.total_learned_buses: int = 0
        self.total_upstream_learned: int = 0
        self.cumulative_abs_error: float = 0.0
        self.within_3_count: int = 0
        self.eval_count: int = 0
        self.last_updated_str: str = "초기화 대기 중"

        # Scikit-learn 보조 모델 (설치된 경우 함께 재학습)
        self.sklearn_ridge = None
        self.history_X: List[List[float]] = []
        self.history_y: List[float] = []

        self.load_state()

    def extract_features(
        self,
        eff_weekday: int,
        eff_time: float,
        headway_gap_min: float = 12.0,
        is_double_decker: bool = False
    ) -> List[float]:
        """
        요일(월~일), 시간(05:00~24:40), 배차간격(Headway), 불금 심야 여부를
        24차원 연속형/커널 피처 벡터로 변환합니다.
        """
        # 1) 13개 시간대 RBF 커널 (각 피크 시간대 주변 ±45분 반응)
        rbf_feats = []
        for c in RBF_CENTERS:
            dist = eff_time - c
            rbf_feats.append(math.exp(-(dist * dist) / (2.0 * (0.75 ** 2))))

        # 2) 주기적 시간 인코딩 (24시간 sin/cos)
        rad = 2.0 * math.pi * (eff_time / 24.0)
        sin_t = math.sin(rad)
        cos_t = math.cos(rad)

        # 3) 요일 원핫/그룹 인코딩 (월화 / 수 / 목 / 금 / 토 / 일)
        is_mon_tue = 1.0 if eff_weekday in [0, 1] else 0.0
        is_wed = 1.0 if eff_weekday == 2 else 0.0
        is_thu = 1.0 if eff_weekday == 3 else 0.0
        is_fri = 1.0 if eff_weekday == 4 else 0.0
        is_sat = 1.0 if eff_weekday == 5 else 0.0
        is_sun = 1.0 if eff_weekday == 6 else 0.0

        # 4) 핵심 상호작용 피처: 불금 심야 피크 & 배차 간격 비율 & 2층 버스 여부
        is_fri_night = 1.0 if (eff_weekday == 4 and eff_time >= 21.5) else 0.0
        headway_norm = max(0.4, min(2.5, headway_gap_min / 12.0)) - 1.0  # 12분 기준 정규화 (-0.6 ~ +1.5)
        dd_feat = 1.0 if is_double_decker else 0.0

        feats = (
            rbf_feats
            + [sin_t, cos_t, is_mon_tue, is_wed, is_thu, is_fri, is_sat, is_sun, is_fri_night, headway_norm, dd_feat]
        )
        return feats[:self.num_features]

    def _dot_predict_delta(self, feats: List[float]) -> float:
        """온라인 가중치 기반 사전(Prior) 대비 보정량 예측"""
        if self.sklearn_ridge is not None and HAS_SKLEARN and len(self.history_y) >= 20:
            try:
                pred = float(self.sklearn_ridge.predict([feats])[0])
                return max(-25.0, min(30.0, pred))
            except Exception:
                pass
        val = self.bias + sum(w * x for w, x in zip(self.weights, feats))
        return max(-25.0, min(30.0, val))

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
        """
        [핵심] 사당역에서 버스가 출발할 때마다 호출되어 실시간으로 모델을 학습(Online Update)시킵니다.
        """
        with MODEL_LOCK:
            weekday = now_dt.weekday()
            eff_weekday, eff_time = get_effective_day_and_time(weekday, now_dt.hour, now_dt.minute)
            slot_idx = int(eff_time * 2)  # 30분 단위 슬롯 (0 ~ 50)
            bucket_key = f"{eff_weekday}_{slot_idx}"

            # 1. 만석 절단(Right-Censoring) 보정:
            # 잔여석이 0석으로 출발했다면 실제 정류장 대기 인원은 버스 좌석수(45/71)보다 많았을 확률이 높음
            effective_demand = float(actual_boarded)
            if remain_seats == 0:
                censor_bonus = 8.0 if crowd_status == "crowded" else 5.0
                effective_demand += censor_bonus

            # 2. 학습 전 현재 모델의 예측치와 실제 탑승객 간 오차(MAE) 평가 기록
            current_pred, _, _ = self._predict_internal(
                now_dt=now_dt,
                heuristic_base=heuristic_prior,
                headway_gap_min=headway_gap_min,
                is_double_decker=is_double_decker
            )
            abs_err = abs(current_pred - actual_boarded)
            self.eval_count += 1
            self.cumulative_abs_error += abs_err
            if abs_err <= 4:
                self.within_3_count += 1

            # 3. 요일×30분 시간대 Empirical Bayes 버킷 업데이트
            b_rec = self.bucket_stats.get(bucket_key, {"count": 0, "mean": float(heuristic_prior), "ewma": float(heuristic_prior)})
            c = b_rec["count"] + 1
            new_mean = b_rec["mean"] + (effective_demand - b_rec["mean"]) / c
            ewma_alpha = 0.35 if c > 1 else 1.0
            new_ewma = (1.0 - ewma_alpha) * b_rec["ewma"] + ewma_alpha * effective_demand
            self.bucket_stats[bucket_key] = {
                "count": c,
                "mean": round(new_mean, 2),
                "ewma": round(new_ewma, 2),
                "last_actual": actual_boarded
            }

            # 4. 온라인 경사하강법(SGD) 가중치 즉시 업데이트 (Prior 대비 잔차 학습)
            target_delta = effective_demand - float(heuristic_prior)
            feats = self.extract_features(eff_weekday, eff_time, headway_gap_min, is_double_decker)
            pred_delta = sum(w * x for w, x in zip(self.weights, feats)) + self.bias
            err = target_delta - pred_delta

            # L2 정규화를 포함한 온라인 SGD 스텝
            lr = self.learning_rate / (1.0 + 0.002 * self.total_learned_buses)
            for i in range(self.num_features):
                grad = -err * feats[i] + 0.01 * self.weights[i]
                self.weights[i] = max(-15.0, min(15.0, self.weights[i] - lr * grad))
            self.bias = max(-15.0, min(15.0, self.bias + lr * err * 0.5))

            # Scikit-learn Ridge 보조 모델 학습 (20건 이상 누적 시 고속 Fit)
            self.history_X.append(feats)
            self.history_y.append(target_delta)
            if len(self.history_y) > 2000:
                self.history_X = self.history_X[-2000:]
                self.history_y = self.history_y[-2000:]
            if HAS_SKLEARN and len(self.history_y) >= 20:
                try:
                    ridge = Ridge(alpha=2.0)
                    ridge.fit(np.array(self.history_X), np.array(self.history_y))
                    self.sklearn_ridge = ridge
                except Exception:
                    pass

            # 5. 실시간 단기 모멘텀(Momentum Residual) 갱신
            now_epoch = time.time()
            instant_residual = effective_demand - float(heuristic_prior)
            if self.last_departure_epoch > 0 and (now_epoch - self.last_departure_epoch < 3600):
                self.ewma_residual = 0.45 * self.ewma_residual + 0.55 * instant_residual
            else:
                self.ewma_residual = instant_residual

            self.last_departure_epoch = now_epoch
            self.total_learned_buses += 1
            self.last_updated_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
            self.last_bus_event = {
                "plate": plate,
                "actualBoarded": actual_boarded,
                "remainSeats": remain_seats,
                "isDoubleDecker": is_double_decker,
                "headwayGapMin": round(headway_gap_min, 1),
                "timeStr": now_dt.strftime("%H:%M"),
                "weekday": eff_weekday,
                "timestamp": now_epoch
            }

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
        """
        상행 4개 정류장(효행초, 아이파크, 신명, 수원대) 통과 버스 실측 소진 좌석수도 온라인 학습합니다.
        """
        with MODEL_LOCK:
            eff_weekday, eff_time = get_effective_day_and_time(now_dt.weekday(), now_dt.hour, now_dt.minute)
            slot_idx = int(eff_time * 2)
            key = f"{station_id}_{eff_weekday}_{slot_idx}"

            rec = self.upstream_bucket_stats.get(key, {"count": 0, "ewma_consumed": float(consumed_seats)})
            c = rec["count"] + 1
            alpha = 0.35 if c > 1 else 1.0
            new_ewma = (1.0 - alpha) * rec["ewma_consumed"] + alpha * float(consumed_seats)
            self.upstream_bucket_stats[key] = {
                "count": c,
                "ewma_consumed": round(new_ewma, 2),
                "last_remain": remain_seats,
                "last_plate": plate,
                "updated_at": now_dt.strftime("%H:%M")
            }
            self.total_upstream_learned += 1
            self.save_state()

    def _predict_internal(
        self,
        now_dt: datetime.datetime,
        heuristic_base: int,
        headway_gap_min: Optional[float] = None,
        is_double_decker: bool = False
    ) -> Tuple[int, str, Dict[str, Any]]:
        """내부 스레드 안전 추론 계산"""
        weekday = now_dt.weekday()
        eff_weekday, eff_time = get_effective_day_and_time(weekday, now_dt.hour, now_dt.minute)
        slot_idx = int(eff_time * 2)
        bucket_key = f"{eff_weekday}_{slot_idx}"

        now_epoch = time.time()
        if headway_gap_min is None:
            if self.last_departure_epoch > 0:
                elapsed_min = (now_epoch - self.last_departure_epoch) / 60.0
                headway_gap_min = max(6.0, min(28.0, elapsed_min))
            else:
                headway_gap_min = 12.0

        # 1. 요일×시간대 버킷 사후 추정치
        b_rec = self.bucket_stats.get(bucket_key)
        bucket_count = b_rec["count"] if b_rec else 0
        bucket_val = b_rec["ewma"] if b_rec else float(heuristic_base)

        # 2. 온라인 회귀 모델 예측치 (Prior + 학습된 Delta)
        feats = self.extract_features(eff_weekday, eff_time, headway_gap_min, is_double_decker)
        reg_delta = self._dot_predict_delta(feats)
        reg_val = float(heuristic_base) + reg_delta

        # 3. 베이즈 가중 융합 (Cold-Start 방어: 학습 샘플 수에 비례하여 ML 가중치 자동 증가)
        # - 전체 학습 대수에 따른 글로벌 신뢰도 (최대 0.80)
        global_alpha = min(0.80, self.total_learned_buses / 80.0)
        # - 해당 요일/시간대 버킷 학습 횟수에 따른 로컬 신뢰도 (최대 0.85)
        bucket_alpha = min(0.85, bucket_count / 6.0)

        if bucket_count > 0:
            ml_combined = 0.65 * bucket_val + 0.35 * reg_val
            effective_alpha = max(global_alpha * 0.6, bucket_alpha)
        else:
            ml_combined = reg_val
            effective_alpha = global_alpha * 0.5

        blended = (1.0 - effective_alpha) * float(heuristic_base) + effective_alpha * ml_combined

        # 4. 실시간 직전 버스 모멘텀 보정 (최근 35분 이내 출발 버스의 실측 오차 지수 감쇄 반영)
        momentum_shift = 0.0
        recent_bus_note = ""
        if self.last_departure_epoch > 0:
            elapsed_min = (now_epoch - self.last_departure_epoch) / 60.0
            if 0 <= elapsed_min <= 35.0:
                decay = math.exp(-elapsed_min / 20.0)
                momentum_shift = max(-10.0, min(14.0, self.ewma_residual * 0.45 * decay))
                if self.last_bus_event:
                    last_p = self.last_bus_event.get("plate", "")
                    last_b = self.last_bus_event.get("actualBoarded", 0)
                    recent_bus_note = f" · 직전({last_p}) {last_b}명 실측 반영"

        final_est = int(round(max(2.0, min(68.0, blended + momentum_shift))))

        ml_meta = {
            "totalLearnedBuses": self.total_learned_buses,
            "bucketLearnedCount": bucket_count,
            "mlBlendWeight": round(effective_alpha * 100, 1),
            "momentumShift": round(momentum_shift, 1),
            "lastBusEvent": self.last_bus_event,
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
        """
        사당역 9번 출구 대기 인원을 [도메인 사전 모델 + 실시간 온라인 학습 ML + 직전 차량 실측 모멘텀]으로 융합 예측합니다.
        """
        with MODEL_LOCK:
            final_est, recent_note, ml_meta = self._predict_internal(
                now_dt=now_dt,
                heuristic_base=heuristic_base,
                headway_gap_min=headway_gap_min,
                is_double_decker=is_double_decker
            )
            if self.total_learned_buses > 0:
                enriched_desc = f"{heuristic_desc} [온라인ML {self.total_learned_buses}대 학습{recent_note}]"
            else:
                enriched_desc = heuristic_desc
            return final_est, enriched_desc, ml_meta

    def get_upstream_learned_consumption(
        self,
        station_id: str,
        now_dt: datetime.datetime,
        default_consumed: int
    ) -> Tuple[int, Optional[str]]:
        """
        상행 정류장(효행초·아이파크·신명·수원대)에 대해 해당 요일·시간대에 학습된 실측 소진 좌석수를 블렌딩합니다.
        """
        with MODEL_LOCK:
            eff_weekday, eff_time = get_effective_day_and_time(now_dt.weekday(), now_dt.hour, now_dt.minute)
            slot_idx = int(eff_time * 2)
            key = f"{station_id}_{eff_weekday}_{slot_idx}"
            rec = self.upstream_bucket_stats.get(key)
            if not rec or rec.get("count", 0) == 0:
                return default_consumed, None

            cnt = rec["count"]
            alpha = min(0.75, cnt / 5.0)
            blended = int(round((1.0 - alpha) * default_consumed + alpha * rec["ewma_consumed"]))
            note = f"실시간 온라인 학습({cnt}회 실측 평균 소진 {int(round(rec['ewma_consumed']))}석) 융합"
            return blended, note

    def get_status_summary(self) -> Dict[str, Any]:
        """프론트엔드 및 모니터링 API용 온라인 학습 상태 요약"""
        mae = round(self.cumulative_abs_error / self.eval_count, 2) if self.eval_count > 0 else None
        hit_rate = round((self.within_3_count / self.eval_count) * 100, 1) if self.eval_count > 0 else None
        global_alpha_pct = int(round(max(15.0, min(80.0, (self.total_learned_buses / 80.0) * 100.0))))
        last_rec = None
        if self.last_bus_event:
            last_rec = {
                "plate": self.last_bus_event.get("plate", ""),
                "boarded": self.last_bus_event.get("actualBoarded", 0),
                "remain": self.last_bus_event.get("remainSeats", 0),
                "timeStr": self.last_bus_event.get("timeStr", "")
            }
        return {
            "mode": "전요일·전시간대 실시간 온라인 학습 (Online SGD + Empirical Bayes)",
            "engine": "Online Incremental SGD + Empirical Bayes Hybrid",
            "coverage": "월~일 전 요일 · 05:00~00:40 전 시간대 실시간 학습",
            "totalSadangLearned": self.total_learned_buses,
            "totalUpstreamLearned": self.total_upstream_learned,
            "totalLearnedAll": self.total_learned_buses + self.total_upstream_learned,
            "sadangTotalLearnedBuses": self.total_learned_buses,
            "upstreamTotalLearnedBuses": self.total_upstream_learned,
            "mlWeightPct": global_alpha_pct,
            "activeTimeBuckets": len(self.bucket_stats),
            "maeSeats": mae,
            "within4SeatsAccuracy": hit_rate,
            "lastUpdated": self.last_updated_str,
            "lastBusEvent": self.last_bus_event,
            "lastSadangRecord": last_rec
        }

    def save_state(self):
        """학습된 가중치와 시간대별 버킷 통계를 디스크에 영구 저장"""
        try:
            state = {
                "bucket_stats": self.bucket_stats,
                "upstream_bucket_stats": self.upstream_bucket_stats,
                "weights": [round(w, 6) for w in self.weights],
                "bias": round(self.bias, 6),
                "ewma_residual": round(self.ewma_residual, 4),
                "last_departure_epoch": self.last_departure_epoch,
                "total_learned_buses": self.total_learned_buses,
                "total_upstream_learned": self.total_upstream_learned,
                "cumulative_abs_error": round(self.cumulative_abs_error, 4),
                "within_3_count": self.within_3_count,
                "eval_count": self.eval_count,
                "last_updated_str": self.last_updated_str,
                "last_bus_event": self.last_bus_event
            }
            with open(MODEL_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] 온라인 학습 모델 저장 오류: {e}")

    def load_state(self):
        """서버 재시작 시 기존 학습된 가중치와 통계 복원"""
        try:
            if not os.path.exists(MODEL_STATE_FILE):
                return
            with open(MODEL_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.bucket_stats = state.get("bucket_stats", {})
            self.upstream_bucket_stats = state.get("upstream_bucket_stats", {})
            loaded_w = state.get("weights", [])
            if len(loaded_w) == self.num_features:
                self.weights = [float(x) for x in loaded_w]
            self.bias = float(state.get("bias", 0.0))
            self.ewma_residual = float(state.get("ewma_residual", 0.0))
            self.last_departure_epoch = float(state.get("last_departure_epoch", 0.0))
            self.total_learned_buses = int(state.get("total_learned_buses", 0))
            self.total_upstream_learned = int(state.get("total_upstream_learned", 0))
            self.cumulative_abs_error = float(state.get("cumulative_abs_error", 0.0))
            self.within_3_count = int(state.get("within_3_count", 0))
            self.eval_count = int(state.get("eval_count", 0))
            self.last_updated_str = state.get("last_updated_str", "복원 완료")
            self.last_bus_event = state.get("last_bus_event", None)
            print(
                f"[OK] 실시간 온라인 학습 모델 복원 완료! "
                f"(사당역 {self.total_learned_buses}대 / 상행 {self.total_upstream_learned}대 누적 학습)"
            )
        except Exception as e:
            print(f"[WARN] 온라인 학습 모델 복원 오류: {e}")


# 싱글톤 인스턴스
online_learner = SadangOnlineLearner()

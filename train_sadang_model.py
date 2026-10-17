"""
[7790 사당역·주요 정류장 실측 텔레메트리 학습 및 성능 리포트 도구]
- 실시간 온라인 학습기(services/online_learner.py)가 버스 통과 시마다 자동 학습한 내역과
  data/sadang_telemetry.csv에 축적된 실측 데이터를 분석하여 MAE 및 적중률 리포트를 출력합니다.
"""
import os
import sys
import csv
from services.online_learner import online_learner
from services.telemetry_logger import SADANG_CSV_PATH, UPSTREAM_CSV_PATH

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_evaluation_report():
    print("=" * 68)
    print("[7790 스마트 통학 AI] 요일별 온라인 AI 학습 상태 리포트")
    print("=" * 68)

    summary = online_learner.get_status_summary()
    print(f"- 학습 엔진      : {summary['engine']}")
    print(f"- 커버리지       : {summary['coverage']}")
    print(f"- 현재 적용 그룹 : {summary.get('currentDayGroupName', '')}")
    print(f"- 사당역 누적 학습: {summary['totalSadangLearned']}대")
    print(f"- 상행 누적 학습  : {summary['totalUpstreamLearned']}대")
    if "dayGroups" in summary:
        print("\n[요일군별 독립 학습 현황]")
        for k, dg in summary["dayGroups"].items():
            print(f"  * {dg['name']}: 사당 {dg['sadangLearned']}대 | 상행 {dg['upstreamLearned']}대 | ML반영률 {dg['mlWeightPct']}%")
    print(f"\n- 활성 시간 버킷  : {summary['activeTimeBuckets']}개 슬롯")
    if summary["maeSeats"] is not None:
        print(f"- 평균 절대 오차  : +-{summary['maeSeats']}석 (MAE)")
        print(f"- +-4석 이내 적중률: {summary['within4SeatsAccuracy']}%")
    print(f"- 최근 학습 시각  : {summary['lastUpdated']}")

    if os.path.exists(SADANG_CSV_PATH):
        with open(SADANG_CSV_PATH, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        print(f"\n[데이터셋 현황] {SADANG_CSV_PATH}: 총 {len(rows)}건 기록됨")
        if rows:
            recent = rows[-5:]
            print("\n최근 기록된 사당역 출발 차량 (최대 5건):")
            for r in recent:
                print(
                    f"  * [{r['timestamp']}] 차량 {r['plate']} | "
                    f"실제 탑승: {r['actual_boarded']}명 (잔여 {r['remain_seats']}/{r['total_seats']}석) | "
                    f"배차간격: {r['headway_gap_min']}분 | 체크포인트: {r['checkpoint_station']}"
                )
    else:
        print(f"\n[안내] 아직 {SADANG_CSV_PATH} 파일에 수집된 데이터가 없습니다. 버스 운행 시 자동 생성됩니다.")
    print("=" * 68)


if __name__ == "__main__":
    run_evaluation_report()

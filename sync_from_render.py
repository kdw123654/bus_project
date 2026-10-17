"""
[Render 클라우드 서버 <-> 로컬 PC 텔레메트리 데이터 양방향 병합 동기화 도구]
- Render 서버에서 실시간 24시간 수집된 CSV와 로컬 PC의 CSV를 중복 없이 합집합(Union) 병합하고,
  온라인 학습 모델(Ridge + SGD)을 재구축한 뒤 실행 중인 로컬 서버 메모리에도 즉시 반영합니다.
"""
import os
import sys
import json
import requests
from core.config import ADMIN_API_TOKEN, REMOTE_SYNC_URL
from services.telemetry_logger import merge_telemetry_csv
from services.online_learner import online_learner

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

RENDER_URL = REMOTE_SYNC_URL.rstrip("/")
DATA_DIR = "data"
MODELS_DIR = "models"


def sync():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    url = f"{RENDER_URL}/api/backup/export"
    print(f"[동기화] Render 서버({url})에서 데이터를 조회합니다...")

    headers = {"User-Agent": "Mozilla/5.0"}
    if ADMIN_API_TOKEN:
        headers["X-Admin-Token"] = ADMIN_API_TOKEN

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if not data.get("success"):
                print(f"[실패] 서버 응답 오류: {data.get('error', '')}")
                return

            sadang_csv = data.get("sadang_csv", "")
            upstream_csv = data.get("upstream_csv", "")

            if sadang_csv:
                cnt_s = merge_telemetry_csv(
                    os.path.join(DATA_DIR, "sadang_telemetry.csv"),
                    sadang_csv,
                    lambda r: (r[0], r[5]),
                    lambda r: r[0]
                )
                print(f"[완료] 사당역 텔레메트리 CSV 병합 완료 (총 {cnt_s}건)")

            if upstream_csv:
                cnt_u = merge_telemetry_csv(
                    os.path.join(DATA_DIR, "upstream_telemetry.csv"),
                    upstream_csv,
                    lambda r: (r[0], r[1], r[6]),
                    lambda r: r[0]
                )
                print(f"[완료] 수원대 상행 텔레메트리 CSV 병합 완료 (총 {cnt_u}건)")

            online_learner.rebuild_from_csvs()
            online_learner.save_state()
            summary = online_learner.get_status_summary()
            print("[완료] 온라인 모델(Ridge + SGD) 재구축 및 JSON 저장 완료")

            # 실행 중인 로컬 서버가 있으면 메모리 상태 즉시 갱신
            try:
                requests.post("http://127.0.0.1:8000/api/ml-reload", headers=headers, timeout=3)
                print("[완료] 실행 중인 로컬 서버(127.0.0.1:8000) 메모리 동기화 완료")
            except Exception:
                pass

            print("\n[동기화 완료 요약]")
            print(f"- 사당역 누적 학습: {summary.get('totalSadangLearned', 0)}대")
            print(f"- 수원대 누적 학습: {summary.get('totalUpstreamLearned', 0)}대")
            print(f"- 총 누적 학습    : {summary.get('totalLearnedAll', 0)}대")
            print(f"- 최근 갱신 시각  : {summary.get('lastUpdated', '')}")
        elif resp.status_code == 404:
            print("[안내] 현재 Render 서버가 이전 버전이라 export 엔드포인트가 아직 없습니다.")
            print("       이번 코드를 Git에 push하여 Render에 배포된 이후부터는 원클릭 동기화가 가능합니다.")
        else:
            print(f"[오류] 응답 코드 {resp.status_code}")
    except Exception as e:
        print(f"[오류] 통신 실패: {e}")


if __name__ == "__main__":
    sync()

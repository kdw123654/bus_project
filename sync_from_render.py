"""
[Render 클라우드 서버 -> 로컬 PC 텔레메트리 데이터 동기화 도구]
- Render 서버에서 실시간 24시간 수집된 CSV와 온라인 모델 상태를 로컬 PC로 원클릭 다운로드하여 동기화합니다.
"""
import os
import sys
import json
import requests

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

RENDER_URL = "https://bus-project-e9nd.onrender.com"
DATA_DIR = "data"
MODELS_DIR = "models"


def sync():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    url = f"{RENDER_URL}/api/backup/export"
    print(f"[동기화] Render 서버({url})에서 데이터를 조회합니다...")

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if not data.get("success"):
                print("[실패] 서버 응답 오류")
                return

            sadang_csv = data.get("sadang_csv", "")
            upstream_csv = data.get("upstream_csv", "")
            model_json = data.get("model_json", {})
            summary = data.get("summary", {})

            if sadang_csv:
                with open(os.path.join(DATA_DIR, "sadang_telemetry.csv"), "w", encoding="utf-8-sig") as f:
                    f.write(sadang_csv)
                print(f"[완료] 사당역 텔레메트리 CSV 동기화 완료")

            if upstream_csv:
                with open(os.path.join(DATA_DIR, "upstream_telemetry.csv"), "w", encoding="utf-8-sig") as f:
                    f.write(upstream_csv)
                print(f"[완료] 수원대 상행 텔레메트리 CSV 동기화 완료")

            if model_json:
                with open(os.path.join(MODELS_DIR, "sadang_online_model.json"), "w", encoding="utf-8") as f:
                    json.dump(model_json, f, ensure_ascii=False, indent=2)
                print(f"[완료] 온라인 모델 JSON 동기화 완료")

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

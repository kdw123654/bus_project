import os
import datetime
from dotenv import load_dotenv

load_dotenv()

# 공공데이터포털 및 GBIS API 키
API_KEY = os.getenv("BUS_API_KEY", "")

# 대한민국 표준시(KST, UTC+9) 보증 헬퍼 (클라우드/도커 배포 환경에서도 KST 기준 동작)
KST = datetime.timezone(datetime.timedelta(hours=9))

def get_kst_now() -> datetime.datetime:
    """항상 대한민국 표준시(KST, UTC+9) 기준의 datetime 객체를 반환합니다."""
    return datetime.datetime.now(KST)

# 노선 및 기준 정류장 ID
STATION_ID = "119000302"         # 사당역 9번 출구 앞 (하행 출발 기점)
ROUTE_REGULAR_ID = "200000149"    # 일반 7790
ROUTE_RESERVE_ID = "233000384"    # 7790 (예약/전세)

# 고성능 인메모리 캐시 TTL (초 단위)
BUS_CACHE_TTL = 6.0          # 6초 도착 정보 캐시 (공공데이터 쿼터 방어)
XAI_CACHE_TTL = 5.0          # 5초 XAI 분석 캐시
TRAFFIC_CACHE_TTL = 12.0     # 12초 도로교통 지연 캐시
WEATHER_CACHE_TTL = 300.0    # 5분 기상 캐시
GBIS_ROUTE_CACHE_TTL = 8.0   # 8초 GBIS 실시간 통합 캐시

# GBIS WAF 차단 방지 브라우저 User-Agent 헤더
GBIS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 5대 핵심 정류장별 XAI 프로필 메타데이터
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

# 7790번 고정 시간표 및 전세버스 안내 정보
TIMETABLE_INFO = {
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

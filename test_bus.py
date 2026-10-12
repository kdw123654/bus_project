import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("BUS_API_KEY")

# 사당역 9번출구(수원대 방면) 정류소 id 및 7790 노선 id
STATION_ID = "119000302"
ROUTE_REGULAR_ID = '200000149' # 7790 일반
ROUTE_RESERVE_ID = '233000384' # 7790 예약/전세

# 도착 정보 URL
url = f"https://apis.data.go.kr/6410000/busarrivalservice/v2/getBusArrivalListv2?serviceKey={API_KEY}&stationId={STATION_ID}&format=json"

print("사당역 7790번 실시간 도착 정보를 조회하는 중...")
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    bus_list = data.get("response", {}).get("msgBody", {}).get("busArrivalList", [])
    
    found = False
    for bus in bus_list:
        route_id = str(bus.get("routeId"))
        route_name = str(bus.get("routeName"))

        # 7790번 버스인지 확인
        if route_id in [ROUTE_REGULAR_ID, ROUTE_RESERVE_ID] or "7790" in route_name:
            found = True

            # 전세/예약버스 여부 판별
            if route_id == ROUTE_RESERVE_ID or "예약" in route_name:
                bus_type = "🔴 [전세/예약 버스 - 탑승 불가 또는 예약자 전용]"
            else:
                bus_type = "🟢 [일반 광역버스 - 일반 버스 이용자 탑승 가능]"

            predict_min = bus.get("predictTime1")
            remain_stations = bus.get("locationNo1")
            remain_seats = bus.get("remainSeatCnt1")
            plate_no = bus.get("plateNo1")

            print("=" * 50)
            print(f"노선: {route_name}번 ({bus_type})")
            print(f"도착 예정: 약 {predict_min}분 후")
            print(f"남은 정류장: {remain_stations}개 전")
            print(f"잔여 좌석: {remain_seats}석")
            print(f"차량 번호: {plate_no}")
            print("=" * 50)

    if not found:
        print("ℹ️ 현재 사당역으로 운행 중인 7790번 버스 도착 정보가 없습니다.")
else:
    print(f"API 호출 실패 (상태 코드: {response.status_code})")
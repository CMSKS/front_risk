import serial
import time

# 시리얼 포트 설정
ser = serial.Serial(
    port='/dev/ttyS0',  # 또는 /dev/ttyAMA0
    baudrate=115200,
    timeout=1
)

def parse_c4001_data(data):
    """C4001 데이터 파싱"""
    if len(data) < 4:
        return None
    
    # 헤더 확인 (보통 0x53, 0x59로 시작)
    if data[0] == 0x53 and data[1] == 0x59:
        # 사람 감지 상태
        if len(data) >= 8:
            target_state = data[2]  # 0: 없음, 1: 움직임, 2: 정지
            movement_dist = data[3] | (data[4] << 8)  # cm
            static_dist = data[5] | (data[6] << 8)    # cm
            
            return {
                'state': target_state,
                'movement_distance': movement_dist,
                'static_distance': static_dist
            }
    return None

try:
    print("C4001 센서 데이터 읽기 시작...")
    buffer = bytearray()
    
    while True:
        if ser.in_waiting > 0:
            # 데이터 읽기
            chunk = ser.read(ser.in_waiting)
            buffer.extend(chunk)
            
            # 헤더 찾기
            while len(buffer) >= 8:
                if buffer[0] == 0x53 and buffer[1] == 0x59:
                    # 패킷 추출
                    packet = buffer[:8]
                    buffer = buffer[8:]
                    
                    # 파싱
                    result = parse_c4001_data(packet)
                    if result:
                        state_msg = ['감지 없음', '움직임 감지', '정지 상태 감지'][result['state']]
                        print(f"상태: {state_msg}")
                        print(f"움직임 거리: {result['movement_distance']} cm")
                        print(f"정지 거리: {result['static_distance']} cm")
                        print("-" * 40)
                else:
                    buffer.pop(0)
        
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n종료합니다.")
finally:
    ser.close()
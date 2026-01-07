#!/usr/bin/env python3
import serial
import time

# 후보 포트들 (Raspberry Pi 5 + Ubuntu 에서 자주 쓰이는 것들)
CANDIDATE_PORTS = [
    "/dev/ttyAMA0",
    "/dev/ttyAMA1",
    "/dev/ttyAMA2",
    "/dev/ttyS0",
    "/dev/ttyS1",
]

BAUD_RATE = 9600   # C4001 기본 속도

def try_port(port: str, baud: int = BAUD_RATE) -> bool:
    print(f"\n[시도] 포트 {port} 에서 센서 데이터 찾는 중...")

    try:
        ser = serial.Serial(port, baudrate=baud, timeout=0.5)
    except Exception as e:
        print(f"  -> 포트 열기 실패: {e}")
        return False

    try:
        # 이 포트에서 잠깐 동안 데이터 들어오는지 확인
        for _ in range(20):  # 약 2초 정도
            line = ser.readline()
            if line:
                print(f"  ✅ 이 포트에서 데이터 감지됨! ({port})")
                try:
                    text = line.decode(errors="replace").strip()
                except Exception:
                    text = "<decode error>"
                hex_str = " ".join(f"{b:02X}" for b in line)
                print(f"  RAW HEX : {hex_str}")
                print(f"  RAW TXT : {text}")
                print("\n==> C4001 센서는 이 포트에 연결된 것으로 보입니다:", port)
                return True
            time.sleep(0.1)
    finally:
        ser.close()

    print("  -> 이 포트에서는 데이터가 안 들어옴 (센서 아님)")
    return False


def main():
    print("C4001 후보 시리얼 포트 자동 탐색 시작")
    print("센서를 가만히 두지 말고, 앞에서 손 흔들거나 움직여 주세요.\n")

    found = False
    for port in CANDIDATE_PORTS:
        if try_port(port):
            found = True
            break

    if not found:
        print("\n❌ 후보 포트들에서 센서 데이터가 발견되지 않았습니다.")
        print("  - 센서 전원(5V / GND) 확인")
        print("  - TX/RX 배선 확인 (센서 TX -> Pi RX, 센서 RX -> Pi TX)")
        print("  - 센서 DIP 스위치가 UART 모드인지 확인")
        print("  - 통신 속도(9600bps) 그대로인지 확인")


if __name__ == "__main__":
    main()

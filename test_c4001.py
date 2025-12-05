import serial
import time

PORT = '/dev/ttyAMA10'

ser = serial.Serial(PORT, 115200, timeout=1)
print(f"포트 {PORT} 연결!")
print("센서 앞에서 손 흔들어보세요!")
print("-" * 50)

try:
    count = 0
    while True:
        if ser.in_waiting > 0:
            data = ser.read(ser.in_waiting)
            print(f"데이터: {' '.join([f'{b:02X}' for b in data])}")
        else:
            print(f"대기 중... {count}", end='\r')
            count += 1
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n종료")
finally:
    ser.close()
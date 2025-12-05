from picamera2 import Picamera2
import time

print("카메라 2대 초기화 중...")

# 카메라 2대 초기화
cam0 = Picamera2(0)
cam1 = Picamera2(1)

# 설정 (1080p)
config0 = cam0.create_still_configuration(main={"size": (1920, 1080)})
config1 = cam1.create_still_configuration(main={"size": (1920, 1080)})

cam0.configure(config0)
cam1.configure(config1)

# 시작
cam0.start()
cam1.start()

print("카메라 준비 중...")
time.sleep(2)

# 동시 촬영
print("사진 촬영!")
cam0.capture_file("cam0_photo.jpg")
cam1.capture_file("cam1_photo.jpg")

print("✓ 사진 저장 완료!")
print("  - cam0_photo.jpg")
print("  - cam1_photo.jpg")

# 종료
cam0.stop()
cam1.stop()

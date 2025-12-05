from picamera2 import Picamera2
import cv2
import numpy as np

print("카메라 2대 스트리밍 시작...")

cam0 = Picamera2(0)
cam1 = Picamera2(1)

# 640x480으로 설정 (부드러운 스트리밍)
config0 = cam0.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
config1 = cam1.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})

cam0.configure(config0)
cam1.configure(config1)

cam0.start()
cam1.start()

print("스트리밍 중... (q 키로 종료)")

try:
    while True:
        # 프레임 캡처
        frame0 = cam0.capture_array()
        frame1 = cam1.capture_array()
        
        # RGB to BGR
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
        
        # 좌우로 나란히 표시
        combined = np.hstack([frame0, frame1])
        
        # 텍스트 추가
        cv2.putText(combined, "Camera 0", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Camera 1", (650, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Dual Camera', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n종료")

finally:
    cam0.stop()
    cam1.stop()
    cv2.destroyAllWindows()

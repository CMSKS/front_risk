#!/usr/bin/env python3
"""빠른 카메라 실시간 프리뷰 (개선 버전)"""
import subprocess
import numpy as np
import cv2
import time

width = 640
height = 480

def setup_camera():
    """카메라 초기 설정"""
    # 센서 flip 설정
    subprocess.run([
        'v4l2-ctl', '-d', '/dev/v4l-subdev5',
        '--set-ctrl=exposure=800',
        '--set-ctrl=analogue_gain=100',
        '--set-ctrl=digital_gain=512',
        '--set-ctrl=horizontal_flip=1',
        '--set-ctrl=vertical_flip=1'
    ], capture_output=True)

    subprocess.run([
        'media-ctl', '-d', '/dev/media1',
        '--set-v4l2', '"imx219 10-0010":0[fmt:SRGGB10_1X10/640x480]'
    ], capture_output=True)

    subprocess.run([
        'media-ctl', '-d', '/dev/media1',
        '-l', '"csi2":4->"rp1-cfe-csi2_ch0":0[1]'
    ], capture_output=True)

def unpack_10bit_bayer(raw_data, width, height):
    """10비트 베이어 데이터 언팩 (최적화)"""
    num_pixels = width * height

    # NumPy를 사용한 빠른 언패킹
    raw_uint16 = np.frombuffer(raw_data, dtype=np.uint8)

    # 간단한 언팩 (속도 우선)
    pixels = np.zeros(num_pixels, dtype=np.uint16)

    idx = 0
    for i in range(0, len(raw_data) - 4, 5):
        if idx + 4 <= num_pixels:
            pixels[idx] = (raw_data[i] << 2) | ((raw_data[i+4] >> 0) & 0x03)
            pixels[idx+1] = (raw_data[i+1] << 2) | ((raw_data[i+4] >> 2) & 0x03)
            pixels[idx+2] = (raw_data[i+2] << 2) | ((raw_data[i+4] >> 4) & 0x03)
            pixels[idx+3] = (raw_data[i+3] << 2) | ((raw_data[i+4] >> 6) & 0x03)
            idx += 4

    bayer_8bit = (pixels >> 2).astype(np.uint8)
    return bayer_8bit.reshape((height, width))

print("카메라 설정 중...")
setup_camera()
print("✓ 설정 완료\n")

print("=" * 60)
print("  빠른 카메라 프리뷰 (개선 버전)")
print("=" * 60)
print("종료: 'q' 키 또는 Ctrl+C")
print("=" * 60)
print()

# OpenCV 윈도우 생성
try:
    cv2.namedWindow('Camera Live', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Live', 1280, 960)
    use_window = True
    print("✓ 윈도우 모드\n")
except:
    use_window = False
    print("! 파일 모드: /tmp/live.jpg\n")

frame_count = 0
start_time = time.time()
last_fps_time = start_time
fps_frames = 0
current_fps = 0

# 색상 패턴 선택 (RGGB 또는 다른 패턴 시도)
bayer_patterns = [
    (cv2.COLOR_BayerRG2BGR, "RGGB"),
    (cv2.COLOR_BayerBG2BGR, "BGGR"),
    (cv2.COLOR_BayerGR2BGR, "GRBG"),
    (cv2.COLOR_BayerGB2BGR, "GBRG"),
]
pattern_idx = 0  # 기본 패턴
current_pattern = bayer_patterns[pattern_idx]

print(f"베이어 패턴: {current_pattern[1]}")
print("패턴 변경: 'p' 키\n")

try:
    while True:
        loop_start = time.time()

        # 프레임 캡처 (subprocess 사용 - 여전히 병목이지만 단순함)
        result = subprocess.run([
            'v4l2-ctl', '-d', '/dev/video8',
            '--stream-mmap', '--stream-count=1',
            '--stream-to=/tmp/frame.bayer'
        ], capture_output=True, timeout=1)

        if result.returncode == 0:
            # 데이터 읽기
            with open('/tmp/frame.bayer', 'rb') as f:
                raw_data = np.fromfile(f, dtype=np.uint8)

            # 베이어 변환
            bayer_img = unpack_10bit_bayer(raw_data, width, height)

            # 선택된 패턴으로 변환
            bgr_img = cv2.cvtColor(bayer_img, current_pattern[0])

            # 밝기 자동 조정
            mean_brightness = bgr_img.mean()
            if mean_brightness < 80:
                alpha = 3.0
                beta = 40
            elif mean_brightness < 120:
                alpha = 2.0
                beta = 20
            else:
                alpha = 1.5
                beta = 10

            enhanced = cv2.convertScaleAbs(bgr_img, alpha=alpha, beta=beta)

            # FPS 계산
            frame_count += 1
            fps_frames += 1
            now = time.time()

            if now - last_fps_time >= 1.0:
                current_fps = fps_frames / (now - last_fps_time)
                fps_frames = 0
                last_fps_time = now

            # FPS 및 정보 표시
            cv2.putText(enhanced, f'FPS: {current_fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(enhanced, f'Pattern: {current_pattern[1]}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if use_window:
                cv2.imshow('Camera Live', enhanced)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    # 베이어 패턴 변경
                    pattern_idx = (pattern_idx + 1) % len(bayer_patterns)
                    current_pattern = bayer_patterns[pattern_idx]
                    print(f"베이어 패턴 변경: {current_pattern[1]}")
            else:
                cv2.imwrite('/tmp/live.jpg', enhanced)
                print(f"\r프레임 #{frame_count} | FPS: {current_fps:.1f}",
                      end='', flush=True)

        # 루프 속도 제한 없음 (최대한 빠르게)

except KeyboardInterrupt:
    print("\n\n종료 중...")
except Exception as e:
    print(f"\n오류: {e}")
    import traceback
    traceback.print_exc()

if use_window:
    cv2.destroyAllWindows()

elapsed = time.time() - start_time
avg_fps = frame_count / elapsed if elapsed > 0 else 0
print(f"\n총 {frame_count} 프레임 | 평균 FPS: {avg_fps:.1f}")
print("=" * 60)

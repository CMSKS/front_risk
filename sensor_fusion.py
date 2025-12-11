#!/usr/bin/env python3
"""
듀얼 카메라 스티칭 + mmWave 레이더 센서 융합 뷰어

- IMX219 카메라 2대를 GStreamer(libcamerasrc)로 받아서 실시간 파노라마 생성
- /dev/ttyAMA0 에 연결된 mmWave 레이더(C4001 등)에서 RAW 데이터를 읽어서
  거리/속도/에너지 추출
- 레이더 좌표(R^2) -> 이미지 좌표(u, v) 변환 행렬 T_RI 를 이용해
  파노라마 영상 위에 레이더 포인트/ROI/파생량(거리, 속도, 동적/정적, 경로 침범도) 표시
"""

import sys
import signal
import math
import time
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np
import serial

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst  # type: ignore

# GStreamer 초기화
Gst.init(None)


# =====================================
# 0. 카메라 정보 (너가 쓰던 그대로)
# =====================================
CAMERAS = [
    {
        'name': 'Camera 0 (i2c@80000)',
        'device': '/base/axi/pcie@120000/rp1/i2c@80000/imx219@10',
    },
    {
        'name': 'Camera 1 (i2c@88000)',
        'device': '/base/axi/pcie@120000/rp1/i2c@88000/imx219@10',
    }
]

# =====================================
# 0-1. 레이더/AGV 파라미터
# =====================================
RADAR_PORT = "/dev/ttyAMA0"
RADAR_BAUD = 9600

LANE_WIDTH_M = 1.0        # AGV 주행로 폭 [m] (경로 침범 판단용)
RISK_TAU = 3.0            # risk 지표의 시간 스케일 파라미터
STATIC_SPEED_THRES = 0.10  # [m/s] 이하면 정적 물체로 본다 (레이더 속도 기준)


# ============================
# 1. 디버그용 매칭 시각화
# ============================
def draw_matches(img1, kp1, img2, kp2, matches, max_num=50):
    matches_to_draw = sorted(matches, key=lambda m: m.distance)[:max_num]
    dbg = cv.drawMatches(
        img1, kp1, img2, kp2, matches_to_draw, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv.imshow("matches", dbg)


# ===================================
# 2. 특징점 검출 + 디스크립터 + 매칭
# ===================================
def detect_and_match_features(img1, img2,
                              detector_type="sift",
                              ratio_test=0.75):
    if detector_type.lower() == "sift":
        if not hasattr(cv, "SIFT_create"):
            raise RuntimeError("이 OpenCV 빌드에는 SIFT가 없습니다.")
        sift = cv.SIFT_create()
    else:
        raise ValueError("지원하지 않는 detector_type")

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return kp1, kp2, []

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches_knn = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches_knn:
        if m.distance < ratio_test * n.distance:
            good_matches.append(m)

    return kp1, kp2, good_matches


# ==========================
# 3. RANSAC으로 호모그래피
# ==========================
def estimate_homography(kp1, kp2, matches,
                        ransac_thresh=4.0):
    if len(matches) < 4:
        raise RuntimeError("매칭점이 너무 적어서 호모그래피 계산 불가")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, ransac_thresh)
    if H is None or mask is None:
        raise RuntimeError("호모그래피 계산 실패")

    inliers = [matches[i] for i in range(len(matches)) if mask[i] != 0]
    return H, inliers, mask


# ===================================
# 4. 호모그래피 워핑 + 공통 캔버스 계산
# ===================================
def warp_to_common_canvas(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.float32([[0, 0],
                           [w1, 0],
                           [w1, h1],
                           [0, h1]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0],
                           [w2, 0],
                           [w2, h2],
                           [0, h2]]).reshape(-1, 1, 2)

    warped_corners2 = cv.perspectiveTransform(corners2, H)

    all_corners = np.concatenate((corners1, warped_corners2), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = [-x_min, -y_min]
    T = np.array([[1, 0, translation[0]],
                  [0, 1, translation[1]],
                  [0, 0, 1]], dtype=np.float32)

    pano_w = x_max - x_min
    pano_h = y_max - y_min

    img1_warp = cv.warpPerspective(img1, T, (pano_w, pano_h))
    img2_warp = cv.warpPerspective(img2, T @ H, (pano_w, pano_h))

    mask1 = np.full((h1, w1), 255, np.uint8)
    mask2 = np.full((h2, w2), 255, np.uint8)

    mask1_warp = cv.warpPerspective(mask1, T, (pano_w, pano_h))
    mask2_warp = cv.warpPerspective(mask2, T @ H, (pano_w, pano_h))

    return img1_warp, img2_warp, mask1_warp, mask2_warp, (pano_w, pano_h)


# =========================================
# 5. 간단 블렌딩
# =========================================
def simple_blend(img1_warp, img2_warp, mask1_warp, mask2_warp):
    pano_h, pano_w = img1_warp.shape[:2]
    pano = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)

    m1 = mask1_warp > 0
    m2 = mask2_warp > 0
    overlap = m1 & m2
    only1 = m1 & (~overlap)
    only2 = m2 & (~overlap)

    pano[only1] = img1_warp[only1]
    pano[only2] = img2_warp[only2]

    if np.any(overlap):
        pano[overlap] = (
            0.5 * img1_warp[overlap].astype(np.float32)
            + 0.5 * img2_warp[overlap].astype(np.float32)
        ).astype(np.uint8)

    return pano


# =========================================
# 6. 전체 스티칭 파이프라인
#    (추가: 파노라마 크기도 리턴)
# =========================================
def stitch_two_images(img1, img2, debug=False):
    kp1, kp2, matches = detect_and_match_features(img1, img2)

    if debug:
        print(f"총 매칭 수: {len(matches)}")
        if len(matches) > 0:
            draw_matches(img1, kp1, img2, kp2, matches)

    if len(matches) < 4:
        raise RuntimeError("유효한 매칭이 부족합니다.")

    H, inliers, _ = estimate_homography(kp1, kp2, matches)

    if debug:
        print(f"RANSAC 인라이어 수: {len(inliers)}")

    img1_warp, img2_warp, mask1_warp, mask2_warp, pano_size = \
        warp_to_common_canvas(img1, img2, H)

    pano = simple_blend(img1_warp, img2_warp, mask1_warp, mask2_warp)
    return pano, pano_size


# =========================================
# 7. GStreamer 파이프라인 + appsink
# =========================================
def create_gst_pipeline(camera_device, width=640, height=480, sink_name="sink"):
    pipeline_desc = (
        f"libcamerasrc camera-name={camera_device} ! "
        f"video/x-raw,width={width},height={height},format=NV21 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        f"appsink name={sink_name} max-buffers=1 drop=true sync=false"
    )
    pipeline = Gst.parse_launch(pipeline_desc)
    if pipeline is None:
        raise RuntimeError("GStreamer 파이프라인 생성 실패")

    sink = pipeline.get_by_name(sink_name)
    if sink is None:
        raise RuntimeError("appsink를 찾을 수 없습니다")

    return pipeline, sink


def gst_sample_to_ndarray(sample):
    buf = sample.get_buffer()
    caps = sample.get_caps()
    s = caps.get_structure(0)
    width = s.get_value('width')
    height = s.get_value('height')

    success, map_info = buf.map(Gst.MapFlags.READ)
    if not success:
        return None

    try:
        data = map_info.data
        frame = np.ndarray(
            (height, width, 3),
            dtype=np.uint8,
            buffer=data
        )
        return frame.copy()
    finally:
        buf.unmap(map_info)


# =========================================
# 8. 레이더 데이터 구조 및 스레드
# =========================================
@dataclass
class RadarDetection:
    """레이더 한 타겟에 대한 정보"""
    r_m: float          # 레이더와의 거리 [m]
    v_mps: float        # 레이더 기준 속도 [m/s]
    x_m: float          # 레이더 좌표계에서 좌우 [m] (왼쪽 음수, 오른쪽 양수)
    y_m: float          # 레이더 좌표계에서 전후 [m] (앞이 양수)
    amplitude: float    # 반사 강도 또는 에너지
    det_id: int = 0     # 옵션: 타겟 ID


class RadarReader(threading.Thread):
    """UART 레이더에서 데이터를 읽어 최신 프레임을 보관하는 스레드"""

    def __init__(self, port: str, baud: int):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: Optional[Tuple[float, List[RadarDetection]]] = None

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.5)
        except Exception as e:
            print(f"[RadarReader] 시리얼 오픈 실패: {e}")
            return

        print(f"[RadarReader] 포트 {self.port}, {self.baud}bps 시작")

        while not self._stop_flag.is_set():
            try:
                line = ser.readline()
                if not line:
                    continue

                ts = time.monotonic()
                detections = self.parse_line(line)
                if detections:
                    with self._lock:
                        self._latest_frame = (ts, detections)

            except Exception as e:
                print(f"[RadarReader] 읽기 에러: {e}")
                continue

        ser.close()
        print("[RadarReader] 종료")

    def stop(self):
        self._stop_flag.set()

    def get_latest_frame(self) -> Optional[Tuple[float, List[RadarDetection]]]:
        with self._lock:
            return self._latest_frame

    @staticmethod
    def parse_line(line: bytes) -> List[RadarDetection]:
        """
        예시 RAW:
          $DFDMD,1,1,1.496,0.100,24601, , *
          0: $DFDMD
          1: flag1
          2: flag2
          3: distance(m)
          4: speed(m/s)
          5: energy
        실제 사용하는 레이더의 바이트 포맷(0x81 헤더 등)이 있다면
        여기에서 바이트 단위로 파싱해서 x_m, y_m, r_m 등을 채우면 됨.
        """
        try:
            text = line.decode("ascii", errors="ignore").strip()
        except Exception:
            return []

        if not text:
            return []

        if "*" in text:
            body, _ = text.split("*", 1)
        else:
            body = text

        if not body.startswith("$DFDMD"):
            # 다른 타입 문장은 일단 무시
            return []

        parts = body.split(",")
        if len(parts) < 6:
            return []

        try:
            distance_m = float(parts[3])
            speed_mps = float(parts[4])
            energy = float(parts[5])
        except Exception:
            return []

        # 현재는 레이더가 각도를 안 주므로, 정면(x=0, y=distance)이라고 가정
        x_m = 0.0
        y_m = max(distance_m, 0.0)

        det = RadarDetection(
            r_m=distance_m,
            v_mps=speed_mps,
            x_m=x_m,
            y_m=y_m,
            amplitude=energy,
            det_id=0,
        )
        return [det]


# =========================================
# 9. 보정 행렬 T_RI 로딩 및 투영 함수
# =========================================
def load_T_RI(path: str = "T_RI.npy") -> np.ndarray:
    """
    레이더 좌표계 (x_gamma, y_gamma, 1) -> 파노라마 이미지 (u, v, 1)
    로 보내는 3x3 행렬을 로딩. 없으면 항등행렬 사용.
    """
    try:
        T = np.load(path).astype(np.float32)
        if T.shape != (3, 3):
            raise ValueError
        print(f"[T_RI] {path} 로딩 완료:")
        print(T)
        return T
    except Exception:
        print("[T_RI] T_RI.npy 를 찾지 못해 항등행렬 사용 (테스트용)")
        return np.eye(3, dtype=np.float32)


def project_radar_to_image(det: RadarDetection, T_RI: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    레이더 좌표 (x_m, y_m) -> 이미지 좌표 (u, v)
    [u, v, 1]^T ~ T_RI [x_m, y_m, 1]^T
    """
    pt = np.array([det.x_m, det.y_m, 1.0], dtype=np.float32)
    uvw = T_RI @ pt
    w = float(uvw[2])
    if abs(w) < 1e-6:
        return None
    u = float(uvw[0] / w)
    v = float(uvw[1] / w)
    return int(round(u)), int(round(v))


# =========================================
# 10. ROI 크기 결정 & 파생량 계산
# =========================================
def compute_roi_size(distance_m: float,
                     base_size_px: int = 160,
                     min_size_px: int = 40,
                     max_size_px: int = 260) -> Tuple[int, int]:
    """
    물체가 가까울수록 ROI를 크게, 멀수록 작게.
    아주 대충: size ~ base / d
    """
    d = max(distance_m, 0.5)
    scale = 1.0 / d
    size = int(base_size_px * scale)
    size = max(min_size_px, min(max_size_px, size))
    return size, size  # 정사각형 ROI


def compute_derived_quantities(det: RadarDetection) -> dict:
    """
    파생량 계산:
      - obstacle_present
      - d_radar
      - height_est (카메라 기반 추정은 여기서는 스킵, placeholder)
      - v_radar
      - in_lane
      - t_collision
      - risk
      - dynamic / static
    """
    d = det.y_m  # 전방 거리 [m] (정면이라고 가정)
    v = det.v_mps

    obstacle_present = det.r_m > 0.0

    # 레인 침범 여부 (x 좌표로 판단)
    in_lane = abs(det.x_m) < (LANE_WIDTH_M / 2.0)

    # 충돌 시간 및 risk
    t_collision = None
    risk = 0.0
    if v > 0.05 and d > 0.0:
        t_collision = d / v
        risk = math.exp(-t_collision / RISK_TAU)
    else:
        t_collision = None
        risk = 0.0

    # 동적/정적
    is_dynamic = abs(v) >= STATIC_SPEED_THRES

    return {
        "obstacle_present": obstacle_present,
        "d_radar": d,
        "v_radar": v,
        "in_lane": in_lane,
        "t_collision": t_collision,
        "risk": risk,
        "is_dynamic": is_dynamic,
    }


# =========================================
# 11. 메인 비디오 + 레이더 융합 루프
# =========================================
def main_video():
    print("GStreamer 듀얼 카메라 + 실시간 스티칭 + mmWave 융합 시작 준비...")

    cam0_dev = CAMERAS[0]['device']
    cam1_dev = CAMERAS[1]['device']

    # 카메라 파이프라인 준비
    pipeline0, sink0 = create_gst_pipeline(cam0_dev, 640, 480, "sink0")
    pipeline1, sink1 = create_gst_pipeline(cam1_dev, 640, 480, "sink1")

    # 레이더 스레드 시작
    T_RI = load_T_RI("T_RI.npy")
    radar_reader = RadarReader(RADAR_PORT, RADAR_BAUD)
    radar_reader.start()

    # 재생 시작
    pipeline0.set_state(Gst.State.PLAYING)
    pipeline1.set_state(Gst.State.PLAYING)

    print("✅ 두 카메라 파이프라인 PLAYING 상태 진입")
    print("ESC 키를 누르거나 Ctrl+C 로 종료 가능합니다.")

    try:
        while True:
            sample0 = sink0.emit("try-pull-sample", 1_000_000_000)
            sample1 = sink1.emit("try-pull-sample", 1_000_000_000)

            if sample0 is None or sample1 is None:
                print("⚠ 샘플을 가져오지 못했습니다. 계속 시도...")
                continue

            frame0 = gst_sample_to_ndarray(sample0)
            frame1 = gst_sample_to_ndarray(sample1)

            if frame0 is None or frame1 is None:
                print("⚠ 프레임 변환 실패. 계속 시도...")
                continue

            pano = None
            pano_size = None
            try:
                pano, pano_size = stitch_two_images(frame0, frame1, debug=False)
            except Exception as e:
                print("스티칭 실패:", e)

            # ====== 레이더 최신 프레임 가져오기 ======
            radar_frame = radar_reader.get_latest_frame()
            if (pano is not None) and (radar_frame is not None):
                ts_radar, detections = radar_frame
                h_pano, w_pano = pano.shape[:2]

                for det in detections:
                    # 레이더 -> 이미지 좌표
                    uv = project_radar_to_image(det, T_RI)
                    if uv is None:
                        continue
                    u, v = uv

                    if not (0 <= u < w_pano and 0 <= v < h_pano):
                        # 이미지 밖이면 스킵
                        continue

                    # 파생량 계산
                    derived = compute_derived_quantities(det)

                    # ROI 크기
                    roi_w, roi_h = compute_roi_size(det.r_m)
                    x1 = max(u - roi_w // 2, 0)
                    y1 = max(v - roi_h // 2, 0)
                    x2 = min(u + roi_w // 2, w_pano - 1)
                    y2 = min(v + roi_h // 2, h_pano - 1)

                    # ROI 및 포인트 표시
                    color_box = (0, 255, 0) if derived["in_lane"] else (0, 255, 255)
                    cv.rectangle(pano, (x1, y1), (x2, y2), color_box, 2)
                    cv.circle(pano, (u, v), 4, (0, 0, 255), -1)

                    # 텍스트: 거리 / 속도 / risk / 동적 여부
                    lines = []
                    lines.append(f"d={derived['d_radar']:.2f} m")
                    lines.append(f"v={derived['v_radar']:.2f} m/s")
                    if derived["t_collision"] is not None:
                        lines.append(f"tc={derived['t_collision']:.1f}s")
                    lines.append(f"dyn={'Y' if derived['is_dynamic'] else 'N'}")
                    lines.append(f"risk={derived['risk']:.2f}")

                    text_x, text_y = x1 + 5, y1 - 5
                    for line in lines:
                        cv.putText(
                            pano, line,
                            (text_x, max(text_y, 15)),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 255, 0), 1, cv.LINE_AA
                        )
                        text_y += 15

            # 원본 카메라도 같이 보기
            cv.imshow("cam0", frame0)
            cv.imshow("cam1", frame1)
            if pano is not None:
                cv.imshow("pano_fused", pano)

            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    except KeyboardInterrupt:
        print("\n[Ctrl+C 감지] 종료합니다.")
    finally:
        radar_reader.stop()
        radar_reader.join(timeout=1.0)
        pipeline0.set_state(Gst.State.NULL)
        pipeline1.set_state(Gst.State.NULL)
        cv.destroyAllWindows()


def main():
    def signal_handler(sig, frame):
        print("\n[시그널 감지] 종료합니다.")
        cv.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    main_video()


if __name__ == "__main__":
    main()

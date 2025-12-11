#!/usr/bin/env python3
"""
듀얼 카메라 실시간 융합 뷰어 (순수 GStreamer + OpenCV)

- IMX219 카메라 2대를 libcamerasrc로 잡음 (네가 쓰던 camera-name 그대로 사용)
- GStreamer 파이프라인에서 appsink로 프레임을 가져와서
- OpenCV로 특징점 매칭 + 호모그래피 + 워핑 + 간단 블렌딩
- cam0 / cam1 / pano 세 창을 띄움
- 초기 한 번만 H(호모그래피) 계산해서 고정, 필요하면 R 키로 재캘리브레이션
"""

import sys
import signal

import cv2 as cv
import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst  # type: ignore

# GStreamer 초기화
Gst.init(None)


# =====================================
# 0. 카메라 정보 (네 GStreamer 코드 그대로)
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
# 5. 간단 블렌딩 (cv.detail 없이)
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
# 6-A. 전체 스티칭 파이프라인 (매 프레임 버전) - 필요시 디버그용
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

    img1_warp, img2_warp, mask1_warp, mask2_warp, _ = \
        warp_to_common_canvas(img1, img2, H)

    pano = simple_blend(img1_warp, img2_warp, mask1_warp, mask2_warp)
    return pano


# =========================================
# 6-B. 이미 알고 있는 H로 스티칭 (고정 H 버전)
# =========================================
def stitch_with_fixed_homography(img1, img2, H):
    """
    이미 계산된 H를 사용해서 두 이미지를 warp + blend만 수행
    """
    img1_warp, img2_warp, mask1_warp, mask2_warp, _ = \
        warp_to_common_canvas(img1, img2, H)

    pano = simple_blend(img1_warp, img2_warp, mask1_warp, mask2_warp)
    return pano


# =========================================
# 6-C. 초기 한 번만 H 계산하는 함수
# =========================================
def compute_homography_once(img1, img2, debug=False):
    kp1, kp2, matches = detect_and_match_features(img1, img2)

    if debug:
        print(f"초기 캘리브레이션 매칭 수: {len(matches)}")
        if len(matches) > 0:
            draw_matches(img1, kp1, img2, kp2, matches)

    if len(matches) < 4:
        raise RuntimeError("초기 H 계산: 유효한 매칭이 부족합니다.")

    H, inliers, _ = estimate_homography(kp1, kp2, matches)

    if debug:
        print(f"초기 RANSAC 인라이어 수: {len(inliers)}")

    return H


# =========================================
# 7. GStreamer 파이프라인 + appsink
# =========================================
def create_gst_pipeline(camera_device, width=640, height=480, sink_name="sink"):
    """
    libcamerasrc camera-name=... !
      video/x-raw,width=640,height=480,format=NV21 !
      videoconvert !
      video/x-raw,format=BGR !
      appsink name=sink ...
    """
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
    """
    appsink에서 받은 Gst.Sample → numpy 배열(BGR)로 변환
    """
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
        # BGR, 3채널
        frame = np.ndarray(
            (height, width, 3),
            dtype=np.uint8,
            buffer=data
        )
        # 복사해서 반환 (GStreamer 버퍼 라이프타임과 분리)
        return frame.copy()
    finally:
        buf.unmap(map_info)


# =========================================
# 8. 메인 루프
# =========================================
def main_video():
    print("GStreamer 듀얼 카메라 + 실시간 스티칭 시작 준비...")

    cam0_dev = CAMERAS[0]['device']
    cam1_dev = CAMERAS[1]['device']

    # 각 카메라에 대한 파이프라인 + appsink 생성
    pipeline0, sink0 = create_gst_pipeline(cam0_dev, 640, 480, "sink0")
    pipeline1, sink1 = create_gst_pipeline(cam1_dev, 640, 480, "sink1")

    # 재생 시작
    pipeline0.set_state(Gst.State.PLAYING)
    pipeline1.set_state(Gst.State.PLAYING)

    print("✅ 두 카메라 파이프라인 PLAYING 상태로 진입")
    print("ESC: 종료 / R: H 다시 캘리브레이션")

    H_fixed = None   # ← 초기에는 없음, 한 번 계산 후 고정

    try:
        while True:
            # 각 카메라에서 샘플 가져오기 (타임아웃: 1초)
            sample0 = sink0.emit("try-pull-sample", 1_000_000_000)
            sample1 = sink1.emit("try-pull-sample", 1_000_000_000)

            if sample0 is None or sample1 is None:
                print("⚠ 샘플을 가져오지 못했습니다 (None). 계속 시도...")
                continue

            frame0 = gst_sample_to_ndarray(sample0)
            frame1 = gst_sample_to_ndarray(sample1)

            if frame0 is None or frame1 is None:
                print("⚠ 프레임 변환 실패. 계속 시도...")
                continue

            # ---------- 고정 H 로직 ----------
            # 1) 아직 H_fixed가 없으면, 한 번만 계산
            if H_fixed is None:
                try:
                    print("📐 초기 H 캘리브레이션 시도 중...")
                    H_fixed = compute_homography_once(frame0, frame1, debug=False)
                    print("✅ 초기 H 캘리브레이션 완료!")
                except Exception as e:
                    print("초기 H 계산 실패, 다음 프레임에서 다시 시도:", e)
                    # H_fixed 못 구했으면 그냥 원본만 보여주고 넘어감
                    cv.imshow("cam0", frame0)
                    cv.imshow("cam1", frame1)
                    key = cv.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break
                    elif key == ord('r') or key == ord('R'):
                        print("🔄 H 초기화 요청 (이미 None 상태).")
                        H_fixed = None
                    continue

            # 2) H_fixed가 있으면, 이걸로만 warp + blend
            pano = None
            try:
                pano = stitch_with_fixed_homography(frame0, frame1, H_fixed)
            except Exception as e:
                print("고정 H로 스티칭 실패:", e)
                pano = None
            # ---------- 고정 H 로직 끝 ----------

            cv.imshow("cam0", frame0)
            cv.imshow("cam1", frame1)
            if pano is not None:
                cv.imshow("pano", pano)

            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r') or key == ord('R'):
                # H를 다시 잡고 싶을 때 (카메라 위치 바꿨을 때 등)
                print("🔄 H 초기화. 다음 프레임에서 다시 캘리브레이션합니다.")
                H_fixed = None

    except KeyboardInterrupt:
        print("\n[Ctrl+C 감지] 종료합니다.")

    # 정리
    pipeline0.set_state(Gst.State.NULL)
    pipeline1.set_state(Gst.State.NULL)
    cv.destroyAllWindows()


def main():
    # 시그널 핸들러: Ctrl+C 시 그나마 깨끗하게 종료
    def signal_handler(sig, frame):
        print("\n[시그널 감지] 종료합니다.")
        cv.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    main_video()


if __name__ == "__main__":
    main()

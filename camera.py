#!/usr/bin/env python3
"""
라즈베리파이 듀얼 카메라 실시간 융합 뷰어
 - Picamera2로 카메라 2대 캡처
 - OpenCV로 특징점 매칭 + 호모그래피 + 워핑 + 간단 블렌딩
 - cam0 / cam1 / pano 창 출력
"""

import cv2 as cv
import numpy as np
from picamera2 import Picamera2


# ============================
# 0. 유틸: 디버그용 그리기 함수
# ============================
def draw_matches(img1, kp1, img2, kp2, matches, max_num=50):
    matches_to_draw = sorted(matches, key=lambda m: m.distance)[:max_num]
    dbg = cv.drawMatches(
        img1, kp1, img2, kp2, matches_to_draw, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv.imshow("matches", dbg)


# ===================================
# 1. 특징점 검출 + 디스크립터 + 매칭
# ===================================
def detect_and_match_features(
    img1,
    img2,
    detector_type="sift",
    ratio_test=0.75,
):
    """
    img1, img2: BGR 또는 RGB 3채널 이미지 (dtype=uint8)
    """

    # --- 특징점 검출기 생성 ---
    if detector_type.lower() == "sift":
        if not hasattr(cv, "SIFT_create"):
            raise RuntimeError("이 OpenCV 빌드에는 SIFT가 없습니다.")
        detector = cv.SIFT_create()
        use_flann = True
    else:
        raise ValueError("지원하지 않는 detector_type")

    # --- 키포인트 + 디스크립터 추출 ---
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return kp1, kp2, []

    # --- 매칭: SIFT → float 디스크립터 → FLANN 사용 ---
    if use_flann:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv.FlannBasedMatcher(index_params, search_params)

        matches_knn = matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches_knn:
            if m.distance < ratio_test * n.distance:
                good_matches.append(m)
    else:
        good_matches = []

    return kp1, kp2, good_matches


# ==========================
# 2. RANSAC으로 호모그래피
# ==========================
def estimate_homography(kp1, kp2, matches, ransac_thresh=4.0):
    if len(matches) < 4:
        raise RuntimeError("매칭점이 너무 적어서 호모그래피 계산 불가")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, ransac_thresh)
    if H is None or mask is None:
        raise RuntimeError("호모그래피 계산 실패")

    # mask는 0/1 또는 0/255 형태
    inliers = [matches[i] for i in range(len(matches)) if mask[i] != 0]
    return H, inliers, mask


# ===================================
# 3. 호모그래피 워핑 + 공통 캔버스 계산
# ===================================
def warp_to_common_canvas(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 각 이미지의 4개 코너
    corners1 = np.float32(
        [[0, 0], [w1, 0], [w1, h1], [0, h1]]
    ).reshape(-1, 1, 2)
    corners2 = np.float32(
        [[0, 0], [w2, 0], [w2, h2], [0, h2]]
    ).reshape(-1, 1, 2)

    # img2의 코너를 img1 좌표계로 투영
    warped_corners2 = cv.perspectiveTransform(corners2, H)

    # 전체 영역 bounding box
    all_corners = np.concatenate((corners1, warped_corners2), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # 음수 좌표가 생기지 않도록 평행이동
    translation = [-x_min, -y_min]
    T = np.array(
        [
            [1, 0, translation[0]],
            [0, 1, translation[1]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    pano_w = x_max - x_min
    pano_h = y_max - y_min

    # 워핑
    img1_warp = cv.warpPerspective(img1, T, (pano_w, pano_h))
    img2_warp = cv.warpPerspective(img2, T @ H, (pano_w, pano_h))

    # 각 이미지의 유효 영역 마스크
    mask1 = np.full((h1, w1), 255, np.uint8)
    mask2 = np.full((h2, w2), 255, np.uint8)

    mask1_warp = cv.warpPerspective(mask1, T, (pano_w, pano_h))
    mask2_warp = cv.warpPerspective(mask2, T @ H, (pano_w, pano_h))

    return img1_warp, img2_warp, mask1_warp, mask2_warp, (pano_w, pano_h)


# =========================================
# 4. 간단 블렌딩 (cv.detail 없이)
# =========================================
def simple_blend(img1_warp, img2_warp, mask1_warp, mask2_warp):
    """
    겹치는 영역은 0.5 / 0.5 평균, 한쪽만 있는 영역은 그대로 사용.
    """
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
# 5. 전체 파이프라인: 두 이미지 스티칭
# =========================================
def stitch_two_images(img1, img2, debug=False):
    """
    img1, img2: 같은 장면을 보는 두 카메라의 프레임 (BGR 또는 RGB)
    """
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
    return pano


# =========================================
# 6. 라즈베리파이용 메인 루프 (Picamera2 두 대)
# =========================================
def main_video():
    print("Picamera2 듀얼 카메라 초기화 중...")

    # 카메라 목록 확인 (IndexError 방지용)
    info = Picamera2.global_camera_info()
    print("감지된 카메라 목록:", info)

    if len(info) < 2:
        print("❌ 감지된 카메라 개수가 2개 미만입니다.")
        print(" - libcamera-hello --list-cameras 로 실제 카메라 인식 상태를 먼저 확인하세요.")
        return

    # 0번, 1번 카메라 오픈
    cam0 = Picamera2(0)
    cam1 = Picamera2(1)

    # 해상도/포맷 설정
    config0 = cam0.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    config1 = cam1.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )

    cam0.configure(config0)
    cam1.configure(config1)

    cam0.start()
    cam1.start()

    print("실시간 스티칭 시작 (ESC로 종료)")

    while True:
        # Picamera2는 RGB888 배열을 반환
        frame0 = cam0.capture_array()
        frame1 = cam1.capture_array()

        # 필요하면 BGR로 바꾸고 싶을 때:
        # frame0_bgr = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        # frame1_bgr = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)
        # 여기서는 그대로 사용 (SIFT에는 큰 영향 없음)
        frame0_use = frame0
        frame1_use = frame1

        pano = None
        try:
            pano = stitch_two_images(frame0_use, frame1_use, debug=False)
        except Exception as e:
            # 매 프레임 완벽할 필요는 없으니, 실패하면 그 프레임은 스킵
            print("스티칭 실패:", e)

        cv.imshow("cam0", frame0_use)
        cv.imshow("cam1", frame1_use)
        if pano is not None:
            cv.imshow("pano", pano)

        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cam0.stop()
    cam1.stop()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main_video()

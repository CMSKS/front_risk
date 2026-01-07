import cv2 as cv
import numpy as np


# ============================
# 0. 유틸: 디버그용 그리기 함수
# ============================
def draw_matches(img1, kp1, img2, kp2, matches, max_num=50):
    matches_to_draw = sorted(matches, key=lambda m: m.distance)[:max_num]
    dbg = cv.drawMatches(img1, kp1, img2, kp2, matches_to_draw, None,
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("matches", dbg)


# ===================================
# 1. 특징점 검출 + 디스크립터 + 매칭
# ===================================
def detect_and_match_features(img1, img2,
                              detector_type="sift",
                              ratio_test=0.75):
    if detector_type.lower() == "sift":
        sift = cv.SIFT_create()
    else:
        raise ValueError("지원하지 않는 detector_type")

    # 특징점 + 디스크립터
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN 매처 (SIFT는 float 디스크립터라 FLANN 사용)
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
# 2. RANSAC으로 호모그래피
# ==========================
def estimate_homography(kp1, kp2, matches,
                        ransac_thresh=4.0):
    if len(matches) < 4:
        raise RuntimeError("매칭점이 너무 적어서 호모그래피 계산 불가")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, ransac_thresh)

    inliers = [matches[i] for i in range(len(matches)) if mask[i]]
    return H, inliers, mask


# ===================================
# 3. 호모그래피 워핑 + 공통 캔버스 계산
# ===================================
def warp_to_common_canvas(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # img1과 img2의 네 꼭짓점 좌표
    corners1 = np.float32([[0, 0],
                           [w1, 0],
                           [w1, h1],
                           [0, h1]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0],
                           [w2, 0],
                           [w2, h2],
                           [0, h2]]).reshape(-1, 1, 2)

    # img2를 img1 좌표계로 투영
    warped_corners2 = cv.perspectiveTransform(corners2, H)

    # 전체 영역 bounding box
    all_corners = np.concatenate((corners1, warped_corners2), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # 음수 좌표 보정용 translation
    translation = [-x_min, -y_min]
    T = np.array([[1, 0, translation[0]],
                  [0, 1, translation[1]],
                  [0, 0, 1]], dtype=np.float32)

    # 캔버스 크기
    pano_w = x_max - x_min
    pano_h = y_max - y_min

    # 각 이미지와 마스크를 동일한 좌표계로 워핑
    img1_warp = cv.warpPerspective(img1, T, (pano_w, pano_h))
    img2_warp = cv.warpPerspective(img2, T @ H, (pano_w, pano_h))

    mask1 = np.zeros((h1, w1), np.uint8)
    mask1[:] = 255
    mask1_warp = cv.warpPerspective(mask1, T, (pano_w, pano_h))

    mask2 = np.zeros((h2, w2), np.uint8)
    mask2[:] = 255
    mask2_warp = cv.warpPerspective(mask2, T @ H, (pano_w, pano_h))

    # corners(각 이미지의 좌상단 in pano 좌표)
    corner1 = (translation[0], translation[1])
    corner2 = (translation[0], translation[1])  # 둘 다 같은 좌표계

    return (img1_warp, img2_warp,
            mask1_warp, mask2_warp,
            corner1, corner2,
            (pano_w, pano_h))


# =====================================
# 4. 노출/색감 보정 (GainCompensator 사용)
# =====================================
def exposure_compensate(imgs, masks, corners):
    # OpenCV detail ExposureCompensator (GAIN_BLOCKS)
    comp = cv.detail.ExposureCompensator_createDefault(
        cv.detail.ExposureCompensator_GAIN_BLOCKS
    )

    # UMat으로 변환
    imgs_umat = [cv.UMat(i) for i in imgs]
    masks_umat = [cv.UMat(m) for m in masks]

    comp.feed(corners, imgs_umat, masks_umat)

    # 보정 적용
    compensated_imgs = []
    for idx, (img, corner, mask) in enumerate(zip(imgs, corners, masks)):
        img_umat = cv.UMat(img)
        comp.apply(idx, corner, img_umat, mask)
        compensated_imgs.append(img_umat.get())

    return compensated_imgs


# =========================================
# 5. GraphCut 기반 솔기 탐색 (Seam Finder)
# =========================================
def find_seams_graphcut(imgs, masks, corners):
    # GraphCutSeamFinder(color+gradient 비용 사용 예시)
    # 일부 빌드에선 detail_GraphCutSeamFinder 로 노출되기도 함
    try:
        seam_finder = cv.detail_GraphCutSeamFinder(
            cv.detail.GraphCutSeamFinderBase_COST_COLOR_GRAD
        )
    except AttributeError:
        # 다른 이름으로 빌드된 경우 (예: cv.detail.GraphCutSeamFinder)
        seam_finder = cv.detail.GraphCutSeamFinder(
            cv.detail.GraphCutSeamFinderBase_COST_COLOR_GRAD
        )

    # GraphCutSeamFinder는 float 타입의 UMat 이미지 사용 권장
    imgs_f = [cv.UMat(i.astype(np.float32)) for i in imgs]
    masks_u = [cv.UMat(m) for m in masks]

    seam_finder.find(imgs_f, corners, masks_u)

    # mask 업데이트 후 다시 numpy로
    refined_masks = [m.get() for m in masks_u]
    return refined_masks


# =========================================
# 6. MultiBandBlender를 이용한 멀티밴드 블렌딩
# =========================================
def multiband_blend(imgs, masks, corners, pano_size, num_bands=5):
    pano_w, pano_h = pano_size

    blender = cv.detail.MultiBandBlender(try_gpu=False, num_bands=num_bands)
    # 전체 ROI 설정
    blender.prepare(cv.Rect(0, 0, pano_w, pano_h))

    for img, mask, corner in zip(imgs, masks, corners):
        # feed(img, mask, top_left)
        blender.feed(img.astype(np.int16), mask, corner)

    # blend 호출 시 초기 dst/dst_mask는 빈 Mat 전달
    dst = np.zeros((pano_h, pano_w, 3), np.int16)
    dst_mask = np.zeros((pano_h, pano_w), np.uint8)
    result, result_mask = blender.blend(dst, dst_mask)

    # 16비트에서 8비트로 변환
    result = cv.normalize(result, None, 0, 255, cv.NORM_MINMAX)
    result = result.astype(np.uint8)

    return result, result_mask


# =========================================
# 7. 전체 파이프라인: 한 번의 스티칭 함수
# =========================================
def stitch_two_images(img1, img2, debug=False):
    # 2. 특징점 + 매칭
    kp1, kp2, matches = detect_and_match_features(img1, img2)

    if debug:
        print(f"총 매칭 수: {len(matches)}")
        draw_matches(img1, kp1, img2, kp2, matches)

    # 3. 호모그래피 추정 (RANSAC)
    H, inliers, inlier_mask = estimate_homography(kp1, kp2, matches)

    if debug:
        print(f"RANSAC 인라이어 수: {len(inliers)}")

    # 4. 공통 캔버스로 워핑
    (img1_warp, img2_warp,
     mask1_warp, mask2_warp,
     corner1, corner2,
     pano_size) = warp_to_common_canvas(img1, img2, H)

    # 5. 노출/색감 보정
    imgs = [img1_warp, img2_warp]
    masks = [mask1_warp, mask2_warp]
    corners = [corner1, corner2]

    imgs = exposure_compensate(imgs, masks, corners)

    # 6. GraphCut 기반 솔기 탐색
    masks = find_seams_graphcut(imgs, masks, corners)

    # 7. MultiBandBlender로 멀티밴드 블렌딩
    pano, pano_mask = multiband_blend(imgs, masks, corners, pano_size)

    return pano


# =========================================
# 8. 라즈베리파이용 메인 루프 (카메라 2대)
# =========================================
def main_video():
    # /dev/video0, /dev/video1 에 연결되어 있다고 가정
    cap0 = cv.VideoCapture(0)
    cap1 = cv.VideoCapture(1)

    if not cap0.isOpened() or not cap1.isOpened():
        print("카메라 열기 실패")
        return

    # 필요 시 해상도 고정 (예: 640x480)
    cap0.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap0.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap1.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap1.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print("프레임 수신 실패")
            break

        try:
            pano = stitch_two_images(frame0, frame1, debug=False)
        except Exception as e:
            print("스티칭 실패:", e)
            pano = None

        # 디버그 출력
        cv.imshow("cam0", frame0)
        cv.imshow("cam1", frame1)
        if pano is not None:
            cv.imshow("pano", pano)

        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap0.release()
    cap1.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main_video()

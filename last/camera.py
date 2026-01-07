#!/usr/bin/env python3
"""
\ub4c0\uc5bc \uce74\uba54\ub77c \uc2e4\uc2dc\uac04 \uc735\ud569 \ubdf0\uc5b4 (\uc21c\uc218 GStreamer + OpenCV)

- IMX219 \uce74\uba54\ub77c 2\ub300\ub97c libcamerasrc\ub85c \uc7a1\uc74c (\ub124\uac00 \uc4f0\ub358 camera-name \uadf8\ub300\ub85c \uc0ac\uc6a9)
- GStreamer \ud30c\uc774\ud504\ub77c\uc778\uc5d0\uc11c appsink\ub85c \ud504\ub808\uc784\uc744 \uac00\uc838\uc640\uc11c
- OpenCV\ub85c \ud2b9\uc9d5\uc810 \ub9e4\uce6d + \ud638\ubaa8\uadf8\ub798\ud53c + \uc6cc\ud551 + \uac04\ub2e8 \ube14\ub80c\ub529
- cam0 / cam1 / pano \uc138 \ucc3d\uc744 \ub744\uc6c0
"""

import sys
import signal

import cv2 as cv
import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst  # type: ignore

# GStreamer \ucd08\uae30\ud654
Gst.init(None)


# =====================================
# 0. \uce74\uba54\ub77c \uc815\ubcf4 (\ub124 GStreamer \ucf54\ub4dc \uadf8\ub300\ub85c)
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
# 1. \ub514\ubc84\uadf8\uc6a9 \ub9e4\uce6d \uc2dc\uac01\ud654
# ============================
def draw_matches(img1, kp1, img2, kp2, matches, max_num=50):
    matches_to_draw = sorted(matches, key=lambda m: m.distance)[:max_num]
    dbg = cv.drawMatches(
        img1, kp1, img2, kp2, matches_to_draw, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv.imshow("matches", dbg)


# ===================================
# 2. \ud2b9\uc9d5\uc810 \uac80\ucd9c + \ub514\uc2a4\ud06c\ub9bd\ud130 + \ub9e4\uce6d
# ===================================
def detect_and_match_features(img1, img2,
                              detector_type="sift",
                              ratio_test=0.75):
    if detector_type.lower() == "sift":
        if not hasattr(cv, "SIFT_create"):
            raise RuntimeError("\uc774 OpenCV \ube4c\ub4dc\uc5d0\ub294 SIFT\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.")
        sift = cv.SIFT_create()
    else:
        raise ValueError("\uc9c0\uc6d0\ud558\uc9c0 \uc54a\ub294 detector_type")

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
# 3. RANSAC\uc73c\ub85c \ud638\ubaa8\uadf8\ub798\ud53c
# ==========================
def estimate_homography(kp1, kp2, matches,
                        ransac_thresh=4.0):
    if len(matches) < 4:
        raise RuntimeError("\ub9e4\uce6d\uc810\uc774 \ub108\ubb34 \uc801\uc5b4\uc11c \ud638\ubaa8\uadf8\ub798\ud53c \uacc4\uc0b0 \ubd88\uac00")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, ransac_thresh)
    if H is None or mask is None:
        raise RuntimeError("\ud638\ubaa8\uadf8\ub798\ud53c \uacc4\uc0b0 \uc2e4\ud328")

    inliers = [matches[i] for i in range(len(matches)) if mask[i] != 0]
    return H, inliers, mask


# ===================================
# 4. \ud638\ubaa8\uadf8\ub798\ud53c \uc6cc\ud551 + \uacf5\ud1b5 \uce94\ubc84\uc2a4 \uacc4\uc0b0
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
# 5. \uac04\ub2e8 \ube14\ub80c\ub529 (cv.detail \uc5c6\uc774)
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
# 6. \uc804\uccb4 \uc2a4\ud2f0\uce6d \ud30c\uc774\ud504\ub77c\uc778
# =========================================
def stitch_two_images(img1, img2, debug=False):
    kp1, kp2, matches = detect_and_match_features(img1, img2)

    if debug:
        print(f"\ucd1d \ub9e4\uce6d \uc218: {len(matches)}")
        if len(matches) > 0:
            draw_matches(img1, kp1, img2, kp2, matches)

    if len(matches) < 4:
        raise RuntimeError("\uc720\ud6a8\ud55c \ub9e4\uce6d\uc774 \ubd80\uc871\ud569\ub2c8\ub2e4.")

    H, inliers, _ = estimate_homography(kp1, kp2, matches)

    if debug:
        print(f"RANSAC \uc778\ub77c\uc774\uc5b4 \uc218: {len(inliers)}")

    img1_warp, img2_warp, mask1_warp, mask2_warp, pano_size = \
        warp_to_common_canvas(img1, img2, H)

    pano = simple_blend(img1_warp, img2_warp, mask1_warp, mask2_warp)
    return pano


# =========================================
# 7. GStreamer \ud30c\uc774\ud504\ub77c\uc778 + appsink
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
        raise RuntimeError("GStreamer \ud30c\uc774\ud504\ub77c\uc778 \uc0dd\uc131 \uc2e4\ud328")

    sink = pipeline.get_by_name(sink_name)
    if sink is None:
        raise RuntimeError("appsink\ub97c \ucc3e\uc744 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4")

    return pipeline, sink


def gst_sample_to_ndarray(sample):
    """
    appsink\uc5d0\uc11c \ubc1b\uc740 Gst.Sample \u2192 numpy \ubc30\uc5f4(BGR)\ub85c \ubcc0\ud658
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
        # BGR, 3\ucc44\ub110
        frame = np.ndarray(
            (height, width, 3),
            dtype=np.uint8,
            buffer=data
        )
        # \ubcf5\uc0ac\ud574\uc11c \ubc18\ud658 (GStreamer \ubc84\ud37c \ub77c\uc774\ud504\ud0c0\uc784\uacfc \ubd84\ub9ac)
        return frame.copy()
    finally:
        buf.unmap(map_info)


# =========================================
# 8. \uba54\uc778 \ub8e8\ud504
# =========================================
def main_video():
    print("GStreamer \ub4c0\uc5bc \uce74\uba54\ub77c + \uc2e4\uc2dc\uac04 \uc2a4\ud2f0\uce6d \uc2dc\uc791 \uc900\ube44...")

    cam0_dev = CAMERAS[0]['device']
    cam1_dev = CAMERAS[1]['device']

    # \uac01 \uce74\uba54\ub77c\uc5d0 \ub300\ud55c \ud30c\uc774\ud504\ub77c\uc778 + appsink \uc0dd\uc131
    pipeline0, sink0 = create_gst_pipeline(cam0_dev, 640, 480, "sink0")
    pipeline1, sink1 = create_gst_pipeline(cam1_dev, 640, 480, "sink1")

    # \uc7ac\uc0dd \uc2dc\uc791
    pipeline0.set_state(Gst.State.PLAYING)
    pipeline1.set_state(Gst.State.PLAYING)

    print("\u2705 \ub450 \uce74\uba54\ub77c \ud30c\uc774\ud504\ub77c\uc778 PLAYING \uc0c1\ud0dc\ub85c \uc9c4\uc785")
    print("ESC \ud0a4\ub97c \ub204\ub974\uba74 \uc885\ub8cc\ud569\ub2c8\ub2e4.")

    try:
        while True:
            # \uac01 \uce74\uba54\ub77c\uc5d0\uc11c \uc0d8\ud50c \uac00\uc838\uc624\uae30 (\ud0c0\uc784\uc544\uc6c3: 1\ucd08)
            sample0 = sink0.emit("try-pull-sample", 1_000_000_000)
            sample1 = sink1.emit("try-pull-sample", 1_000_000_000)

            if sample0 is None or sample1 is None:
                print("\u26a0 \uc0d8\ud50c\uc744 \uac00\uc838\uc624\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4 (None). \uacc4\uc18d \uc2dc\ub3c4...")
                continue

            frame0 = gst_sample_to_ndarray(sample0)
            frame1 = gst_sample_to_ndarray(sample1)

            if frame0 is None or frame1 is None:
                print("\u26a0 \ud504\ub808\uc784 \ubcc0\ud658 \uc2e4\ud328. \uacc4\uc18d \uc2dc\ub3c4...")
                continue

            pano = None
            try:
                pano = stitch_two_images(frame0, frame1, debug=False)
            except Exception as e:
                # \ub9e4 \ud504\ub808\uc784 \uc644\ubcbd\ud560 \ud544\uc694\ub294 \uc5c6\uc73c\ubbc0\ub85c, \uc2e4\ud328\ud558\uba74 \ub118\uc5b4\uac10
                print("\uc2a4\ud2f0\uce6d \uc2e4\ud328:", e)

            cv.imshow("cam0", frame0)
            cv.imshow("cam1", frame1)
            if pano is not None:
                cv.imshow("pano", pano)

            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    except KeyboardInterrupt:
        print("\n[Ctrl+C \uac10\uc9c0] \uc885\ub8cc\ud569\ub2c8\ub2e4.")

    # \uc815\ub9ac
    pipeline0.set_state(Gst.State.NULL)
    pipeline1.set_state(Gst.State.NULL)
    cv.destroyAllWindows()


def main():
    # \uc2dc\uadf8\ub110 \ud578\ub4e4\ub7ec: Ctrl+C \uc2dc \uadf8\ub098\ub9c8 \uae68\ub057\ud558\uac8c \uc885\ub8cc
    def signal_handler(sig, frame):
        print("\n[\uc2dc\uadf8\ub110 \uac10\uc9c0] \uc885\ub8cc\ud569\ub2c8\ub2e4.")
        cv.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    main_video()


if __name__ == "__main__":
    main()

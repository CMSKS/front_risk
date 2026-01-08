#!/usr/bin/env python3
"""
Dual Camera Lane Intrusion Detection System with mmWave + ROS2
카메라 2대 + mmWave 센서를 통합한 위험도 산정 시스템
ROS2를 통해 원격 서버에서 이미지와 알림 수신

[수정 내용]
- 오른쪽 패널(우측 절반)에 팀2 결과 표시 추가
  - 최신 이미지: ~/received_team2_images/{danger,error,safe}/*.jpg
  - 매칭 json:   ~/received_team2_jsons/{danger,error,safe}/{stem}.json
  - json 없으면 해당 클래스 폴더의 최신 json fallback
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import numpy as np
import cv2
import sys
import signal
import threading
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image as PILImage, ImageDraw, ImageFont

# cv2.freetype 초기화
try:
    import cv2.freetype as ft
    FREETYPE_AVAILABLE = True
except ImportError:
    FREETYPE_AVAILABLE = False

# ROS2 imports (선택적)
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object  # 더미 클래스

from mmwave_sensor import MMWaveSensor


class DualCameraLaneDetector(Node):
    def __init__(self):
        """카메라 2대를 사용한 차선 침범 감지 + ROS2 통신 (선택적)"""
        # ROS2 초기화 (사용 가능한 경우)
        if ROS2_AVAILABLE:
            super().__init__('dual_lane_detector')
            print("✓ ROS2 모드 활성화")
        else:
            print("! ROS2 없이 실행 (카메라 모드만)")

        # 카메라 장치 경로
        self.camera_devices = [
            '/base/axi/pcie@120000/rp1/i2c@80000/imx219@10',  # Camera 0
            '/base/axi/pcie@120000/rp1/i2c@88000/imx219@10'   # Camera 1
        ]

        # 프레임 버퍼 (카메라별)
        self.frames = [None, None]
        self.frame_locks = [threading.Lock(), threading.Lock()]

        # 침입 감지 상태
        self.intrusion_detected = False

        # ROS2 관련 변수
        self.server_image_1 = None  # 첫 번째 서버 이미지
        self.server_image_2 = None  # 두 번째 서버 이미지
        self.server_image_lock = threading.Lock()
        self.alert_messages = []
        self.alert_lock = threading.Lock()

        # 알림 X 버튼 클릭 영역 저장 (클릭 감지용)
        self.alert_close_buttons = []  # [(x1, y1, x2, y2, alert_index), ...]
        self.alert_buttons_lock = threading.Lock()

        # ---- Team2 (server postprocessed images) file-based inputs
        self.team2_img_root = Path.home() / "received_team2_images"
        self.team2_json_root = Path.home() / "received_team2_jsons"
        self.team2_classes = ["danger", "error", "safe"]

        self.team2_image_0 = None        # camera_0 이미지
        self.team2_image_1 = None        # camera_1 이미지
        self.team2_json = None           # dict (시간 정보 포함)
        self.team2_last_key = None       # (cls, stem) to avoid reloading every frame
        self.team2_lock = threading.Lock()

        # ROS2 Subscribers (사용 가능한 경우)
        if ROS2_AVAILABLE:
            self.bridge = CvBridge()

            # 두 개의 이미지 토픽 구독 (※현재 UI에서는 팀2 파일 기반 표시를 우선 사용)
            self.image_sub_1 = self.create_subscription(
                Image,
                '/server/camera/image1',
                self.server_image_1_callback,
                10
            )

            self.image_sub_2 = self.create_subscription(
                Image,
                '/server/camera/image2',
                self.server_image_2_callback,
                10
            )

            self.alert_sub = self.create_subscription(
                String,
                '/server/alerts',
                self.alert_callback,
                10
            )

            self.get_logger().info('ROS2 subscribers initialized (2 images + alerts)')

        # mmWave 센서 초기화 (UART)
        print("mmWave 센서 초기화 중...")
        try:
            self.mmwave_sensor = MMWaveSensor(port='/dev/ttyAMA0', baudrate=9600)
            self.mmwave_enabled = True
        except Exception as e:
            print(f"mmWave 센서 초기화 실패: {e}")
            self.mmwave_sensor = None
            self.mmwave_enabled = False

        # 위험도 관련
        self.normalized_risk = 0.0  # 0~1 정규화 위험도
        self.risk_factors = {
            'distance': 0.0,
            'velocity': 0.0,
            'path_intrusion': 0.0,
            'motion_type': 0.0
        }

        # 위험도 계산 가중치
        self.risk_weights = {
            'distance': 0.35,
            'velocity': 0.30,
            'path_intrusion': 0.25,
            'motion_type': 0.10
        }

        # FreeType 폰트 초기화 (한글 지원)
        self.ft_font = None
        if FREETYPE_AVAILABLE:
            try:
                self.ft_font = ft.createFreeType2()
                self.ft_font.loadFontData(fontFileName='/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf', id=0)
                print("✓ FreeType 한글 폰트 로드 성공")
            except Exception as e:
                print(f"FreeType 폰트 로드 실패: {e}")
                self.ft_font = None

        # 거리/속도 임계값
        self.critical_distance = 30.0   # cm
        self.safe_distance = 200.0      # cm
        self.critical_velocity = -50.0  # cm/s
        self.safe_velocity = 10.0       # cm/s
        self.motion_velocity_threshold = 5.0  # cm/s

        # 개별 카메라 크기: 640x480
        # 합친 화면 크기: 640x960 (위아래로 배치)
        self.cam_width = 640
        self.cam_height = 480
        self.total_height = 960

        # 전체화면 설정 (1920x1080)
        self.screen_width = 1920
        self.screen_height = 1080
        self.use_fullscreen = True

        # 연속된 차선 정의 (합쳐진 화면 기준)
        self.left_lane = np.array([[150, 50], [50, 910]], dtype=np.int32)
        self.right_lane = np.array([[490, 50], [590, 910]], dtype=np.int32)

        # 배경 차분기 (카메라별)
        self.bg_subtractors = [
            cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True),
            cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        ]

        # GStreamer 초기화
        Gst.init(None)
        self.pipelines = [None, None]

    # ---------------- Team2 helpers ----------------
    def _find_latest_team2_images(self) -> Tuple[Optional[str], Optional[Path], Optional[Path]]:
        """~/received_team2_images/{danger,error,safe} 에서 최신 camera_0.jpg, camera_1.jpg를 찾음."""
        latest_cls = None
        latest_cam0 = None
        latest_cam1 = None
        latest_mtime = -1.0

        for cls in self.team2_classes:
            d = self.team2_img_root / cls
            if not d.exists():
                continue

            # camera_0.jpg와 camera_1.jpg를 쌍으로 찾음
            for p0 in d.glob("*camera_0.jpg"):
                try:
                    # camera_0의 stem에서 camera_0을 제거하여 base 이름 추출
                    base_name = p0.stem.replace("_camera_0", "")
                    p1 = d / f"{base_name}_camera_1.jpg"

                    # 두 파일이 모두 존재하는지 확인
                    if p1.exists():
                        mt = max(p0.stat().st_mtime, p1.stat().st_mtime)
                        if mt > latest_mtime:
                            latest_mtime = mt
                            latest_cam0 = p0
                            latest_cam1 = p1
                            latest_cls = cls
                except Exception:
                    continue

        return latest_cls, latest_cam0, latest_cam1

    def _load_team2_from_disk_if_needed(self):
        """
        최신 팀2 camera_0.jpg, camera_1.jpg를 확인하고 변경되었으면:
          - 이미지 2개 로드
          - 같은 stem의 json 로드 (시간 정보 포함)
        """
        cls, cam0_path, cam1_path = self._find_latest_team2_images()
        if cam0_path is None or cam1_path is None or cls is None:
            return

        # base_name 추출 (예: "2026.01.08_15_30_45")
        base_name = cam0_path.stem.replace("_camera_0", "")
        key = (cls, base_name)

        with self.team2_lock:
            if self.team2_last_key == key:
                return  # 변경 없음

        # 1) 이미지 2개 로드
        img0 = cv2.imread(str(cam0_path), cv2.IMREAD_COLOR)
        img1 = cv2.imread(str(cam1_path), cv2.IMREAD_COLOR)
        if img0 is None or img1 is None:
            return

        # 2) json 로드 (동일 base_name 우선)
        json_path = (self.team2_json_root / cls / f"{base_name}.json")
        data = None

        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = None
        else:
            # fallback: 해당 클래스 폴더에서 가장 최신 json 하나라도 읽기
            d = self.team2_json_root / cls
            latest_json = None
            latest_mtime = -1.0
            if d.exists():
                for p in d.glob("*.json"):
                    try:
                        mt = p.stat().st_mtime
                        if mt > latest_mtime:
                            latest_mtime = mt
                            latest_json = p
                    except Exception:
                        continue
            if latest_json is not None:
                try:
                    with open(latest_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = None

        with self.team2_lock:
            self.team2_image_0 = img0
            self.team2_image_1 = img1
            self.team2_json = data
            self.team2_last_key = key

    def _parse_timestamp_from_filename(self, base_name):
        """
        파일명에서 시간 정보 추출 (예: "2026.01.08_15_30_45")
        반환: datetime 객체 또는 None
        """
        try:
            # "2026.01.08_15_30_45" -> datetime
            parts = base_name.replace(".", "_").split("_")
            if len(parts) >= 6:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                hour = int(parts[3])
                minute = int(parts[4])
                second = int(parts[5])
                return datetime(year, month, day, hour, minute, second)
        except:
            pass
        return None

    def _format_time_ago(self, dt):
        """
        datetime 객체를 "지금", "N분 전" 형식으로 변환
        """
        if dt is None:
            return "알 수 없음"

        now = datetime.now()
        diff = (now - dt).total_seconds()

        if diff < 0:
            diff = 0

        if diff < 30:
            return "지금"
        elif diff < 60:
            return f"{int(diff)}초 전"
        elif diff < 3600:
            return f"{int(diff // 60)}분 전"
        elif diff < 86400:
            return f"{int(diff // 3600)}시간 전"
        else:
            return f"{int(diff // 86400)}일 전"

    def _draw_team2_panel(self, frame, x1, y1, x2, y2):
        """
        frame 위에 팀2 서버 후처리 이미지 2개를 그림.
        좌표는 (x1,y1) ~ (x2,y2)
        왼쪽: camera_0, 오른쪽: camera_1
        상단에 팝업으로 시간 정보 표시
        """
        # 배경
        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), -1)

        # 최신 파일 로드(필요 시)
        self._load_team2_from_disk_if_needed()

        with self.team2_lock:
            img0 = None if self.team2_image_0 is None else self.team2_image_0.copy()
            img1 = None if self.team2_image_1 is None else self.team2_image_1.copy()
            data = self.team2_json.copy() if isinstance(self.team2_json, dict) else None
            last_key = self.team2_last_key

        panel_w = x2 - x1
        panel_h = y2 - y1

        # 이미지가 없으면 메시지 표시
        if img0 is None or img1 is None:
            self._put_korean_text(frame, "서버 이미지 대기 중...",
                                (x1 + panel_w // 2 - 150, y1 + panel_h // 2),
                                font_size=35, color=(120, 120, 120))
            return

        # === 상단 팝업 (시간 정보) ===
        popup_height = 90
        popup_y1 = y1
        popup_y2 = y1 + popup_height

        # 팝업 배경 (어두운 회색)
        cv2.rectangle(frame, (x1, popup_y1), (x2, popup_y2), (50, 50, 50), -1)
        # 팝업 하단 테두리
        cv2.line(frame, (x1, popup_y2), (x2, popup_y2), (100, 100, 100), 2)

        # 시간 정보 추출
        timestamp_dt = None
        if last_key is not None:
            cls, base_name = last_key
            timestamp_dt = self._parse_timestamp_from_filename(base_name)

        time_ago_str = self._format_time_ago(timestamp_dt)

        # 팝업 텍스트: 왼쪽에 "서버 업데이트", 오른쪽에 "N분 전"
        self._put_korean_text(frame, "서버 업데이트", (x1 + 20, popup_y1 + 30),
                             font_size=32, color=(255, 255, 255))

        # 시간 텍스트 오른쪽 정렬 (대략적인 위치)
        time_x = x2 - 220
        self._put_korean_text(frame, time_ago_str, (time_x, popup_y1 + 30),
                             font_size=30, color=(100, 200, 255))

        # === 이미지 영역 (팝업 아래) ===
        img_start_y = popup_y2 + 20
        img_area_h = y2 - img_start_y - 20
        img_area_w = (panel_w - 60) // 2  # 좌우 2개 이미지

        # 왼쪽: camera_0
        img0_x1 = x1 + 20
        img0_y1 = img_start_y
        self._draw_single_server_image(frame, img0, img0_x1, img0_y1, img_area_w, img_area_h, "Camera 0")

        # 오른쪽: camera_1
        img1_x1 = img0_x1 + img_area_w + 20
        img1_y1 = img_start_y
        self._draw_single_server_image(frame, img1, img1_x1, img1_y1, img_area_w, img_area_h, "Camera 1")

    def _draw_single_server_image(self, frame, img, x, y, w, h, label):
        """
        단일 서버 이미지를 지정된 영역에 그림
        """
        if img is None:
            return

        # 이미지 리사이즈 (비율 유지)
        ih, iw = img.shape[:2]
        scale = min(w / iw, h / ih)
        rw, rh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(img, (rw, rh))

        # 중앙 정렬
        px = x + (w - rw) // 2
        py = y + (h - rh) // 2

        # 이미지 붙이기
        frame[py:py + rh, px:px + rw] = resized

        # 라벨 (이미지 아래) - PIL로 한글/영문 지원
        label_y = py + rh + 5
        self._put_korean_text(frame, label, (px, label_y),
                             font_size=20, color=(200, 200, 200))

    # ---------------- ROS callbacks ----------------
    def server_image_1_callback(self, msg):
        """ROS2 서버 이미지 1 수신 콜백"""
        if not ROS2_AVAILABLE:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.server_image_lock:
                self.server_image_1 = cv_image.copy()
        except Exception as e:
            self.get_logger().error(f'서버 이미지 1 변환 실패: {e}')

    def server_image_2_callback(self, msg):
        """ROS2 서버 이미지 2 수신 콜백"""
        if not ROS2_AVAILABLE:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.server_image_lock:
                self.server_image_2 = cv_image.copy()
        except Exception as e:
            self.get_logger().error(f'서버 이미지 2 변환 실패: {e}')

    def alert_callback(self, msg):
        """ROS2 알림 메시지 수신 콜백"""
        if not ROS2_AVAILABLE:
            return
        try:
            alert_data = json.loads(msg.data)
            with self.alert_lock:
                self.alert_messages = [alert_data]
            self.get_logger().info(f'알림 수신: {alert_data}')
        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON 파싱 실패: {e}')

    # ---------------- camera pipeline ----------------
    def create_pipeline(self, camera_index):
        """개별 카메라의 GStreamer 파이프라인 생성"""
        device = self.camera_devices[camera_index]
        pipeline_str = (
            f'libcamerasrc camera-name={device} ! '
            'video/x-raw,width=640,height=480,format=NV12 ! '
            'videoconvert ! '
            'video/x-raw,format=BGR ! '
            f'appsink name=sink{camera_index} emit-signals=true sync=false max-buffers=2 drop=true'
        )
        print(f"카메라 {camera_index} 파이프라인 생성 중...")

        try:
            pipeline = Gst.parse_launch(pipeline_str)
            appsink = pipeline.get_by_name(f'sink{camera_index}')
            appsink.connect('new-sample', self.on_new_sample, camera_index)
            print(f"✓ 카메라 {camera_index} 파이프라인 생성 완료")
            return pipeline
        except Exception as e:
            print(f"✗ 카메라 {camera_index} 파이프라인 생성 실패: {e}")
            return None

    def on_new_sample(self, appsink, camera_index):
        """새 프레임이 도착했을 때 콜백"""
        sample = appsink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        caps = sample.get_caps()

        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        frame = np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=map_info.data)

        with self.frame_locks[camera_index]:
            self.frames[camera_index] = frame.copy()

        buffer.unmap(map_info)
        return Gst.FlowReturn.OK

    def get_combined_frame(self):
        """2개 카메라 프레임을 세로로 합치기 (위아래)"""
        with self.frame_locks[0]:
            frame0 = self.frames[0].copy() if self.frames[0] is not None else None
        with self.frame_locks[1]:
            frame1 = self.frames[1].copy() if self.frames[1] is not None else None

        if frame0 is None or frame1 is None:
            return None
        return np.vstack([frame0, frame1])

    # ---------------- risk calc ----------------
    def _calculate_distance_risk(self, distance: float) -> float:
        if distance is None or distance <= 0:
            return 0.0
        if distance <= self.critical_distance:
            return 1.0
        if distance >= self.safe_distance:
            return 0.0
        return 1.0 - (distance - self.critical_distance) / (self.safe_distance - self.critical_distance)

    def _calculate_velocity_risk(self, velocity: float) -> float:
        if velocity is None:
            return 0.0
        if velocity >= self.safe_velocity:
            return 0.0
        if velocity <= self.critical_velocity:
            return 1.0
        if velocity > 0:
            return 0.1
        return abs(velocity) / abs(self.critical_velocity)

    def _calculate_path_intrusion_risk(self, intrusion_detected: bool, intrusion_count: int) -> float:
        if not intrusion_detected:
            return 0.0
        base_risk = 0.5
        count_factor = min(intrusion_count * 0.2, 0.5)
        return min(base_risk + count_factor, 1.0)

    def _calculate_motion_type_risk(self, is_dynamic: bool, motion_intensity: float = 0.5) -> float:
        if is_dynamic:
            return 0.3 + (motion_intensity * 0.7)
        return 0.1

    def calculate_comprehensive_risk(self, mmwave_data, intrusion_count):
        detected = mmwave_data.get('detected', False) and mmwave_data.get('connected', False)

        distance = mmwave_data.get('distance')
        distance_risk = self._calculate_distance_risk(distance) if detected else 0.0

        velocity = mmwave_data.get('velocity')
        velocity_risk = self._calculate_velocity_risk(velocity) if detected else 0.0

        path_intrusion_risk = self._calculate_path_intrusion_risk(self.intrusion_detected, intrusion_count)

        is_dynamic = False
        motion_intensity = 0.0
        if detected and velocity is not None:
            is_dynamic = abs(velocity) > self.motion_velocity_threshold
            if is_dynamic:
                max_velocity = abs(self.critical_velocity)
                motion_intensity = min(abs(velocity) / max_velocity, 1.0)

        motion_type_risk = self._calculate_motion_type_risk(is_dynamic, motion_intensity)

        self.risk_factors = {
            'distance': distance_risk,
            'velocity': velocity_risk,
            'path_intrusion': path_intrusion_risk,
            'motion_type': motion_type_risk
        }

        weights = self.risk_weights
        total_weight = sum(weights.values())

        fused_risk = (
            distance_risk * weights['distance'] +
            velocity_risk * weights['velocity'] +
            path_intrusion_risk * weights['path_intrusion'] +
            motion_type_risk * weights['motion_type']
        ) / total_weight

        high_risk_count = sum(1 for risk in self.risk_factors.values() if risk > 0.7)
        if high_risk_count >= 3:
            fused_risk = min(1.0, fused_risk * 1.3)
        elif high_risk_count >= 2:
            fused_risk = min(1.0, fused_risk * 1.15)

        self.normalized_risk = max(0.0, min(1.0, fused_risk))
        return self.normalized_risk

    # ---------------- vision ----------------
    def draw_lanes(self, frame):
        cv2.line(frame, tuple(self.left_lane[0]), tuple(self.left_lane[1]), (0, 255, 0), 3)
        cv2.line(frame, tuple(self.right_lane[0]), tuple(self.right_lane[1]), (0, 255, 0), 3)

        roi_points = np.array([self.left_lane[0], self.right_lane[0], self.right_lane[1], self.left_lane[1]], dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_points], (0, 255, 0))
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        cv2.line(frame, (0, 480), (640, 480), (255, 255, 0), 2)
        return frame

    def detect_intrusion(self, frame):
        if frame is None:
            return False, 0

        frame0 = frame[:480, :]
        frame1 = frame[480:, :]

        fg_mask0 = self.bg_subtractors[0].apply(frame0)
        fg_mask1 = self.bg_subtractors[1].apply(frame1)
        fg_mask = np.vstack([fg_mask0, fg_mask1])

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        roi_points = np.array([self.left_lane[0], self.right_lane[0], self.right_lane[1], self.left_lane[1]], dtype=np.int32)
        cv2.fillPoly(roi_mask, [roi_points], 255)

        intrusion_mask = cv2.bitwise_and(fg_mask, roi_mask)
        contours, _ = cv2.findContours(intrusion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 1000
        intrusion = False
        intrusion_count = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                intrusion = True
                intrusion_count += 1
                cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        return intrusion, intrusion_count

    # ---------------- UI drawing ----------------
    def _put_korean_text(self, img, text, pos, font_size=30, color=(255, 255, 255)):
        """PIL을 사용하여 한글 텍스트 그리기 (최적화 버전)"""
        try:
            text = str(text)

            # 한글 폰트 로드
            font = None
            font_paths = [
                "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            ]

            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except Exception:
                    continue

            if font is None:
                # 폰트 로드 실패
                cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_size/30.0, color, 2)
                return

            # 텍스트 크기 계산을 위한 임시 이미지
            temp_img = PILImage.new('RGB', (1, 1))
            temp_draw = ImageDraw.Draw(temp_img)
            bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # 여유 공간 추가
            pad = 10
            text_width += pad * 2
            text_height += pad * 2

            # 작은 텍스트 이미지 생성 (투명 배경)
            txt_img = PILImage.new('RGBA', (text_width, text_height), (0, 0, 0, 0))
            txt_draw = ImageDraw.Draw(txt_img)

            # BGR -> RGB 색상 변환
            color_rgb = (color[2], color[1], color[0], 255)

            # 텍스트 그리기
            txt_draw.text((pad, pad), text, font=font, fill=color_rgb)

            # RGBA -> BGR 변환
            txt_img_bgr = cv2.cvtColor(np.array(txt_img), cv2.COLOR_RGBA2BGR)

            # 알파 채널 추출
            alpha = np.array(txt_img)[:, :, 3] / 255.0

            # 오버레이 위치 계산
            x, y = pos
            y1 = max(0, y)
            y2 = min(img.shape[0], y + text_height)
            x1 = max(0, x)
            x2 = min(img.shape[1], x + text_width)

            # 실제 오버레이할 영역 크기
            h = y2 - y1
            w = x2 - x1

            if h > 0 and w > 0:
                # 텍스트 이미지에서 해당 영역만 추출
                txt_y1 = 0 if y >= 0 else -y
                txt_x1 = 0 if x >= 0 else -x
                txt_region = txt_img_bgr[txt_y1:txt_y1+h, txt_x1:txt_x1+w]
                alpha_region = alpha[txt_y1:txt_y1+h, txt_x1:txt_x1+w]

                # 알파 블렌딩
                for c in range(3):
                    img[y1:y2, x1:x2, c] = (
                        alpha_region * txt_region[:, :, c] +
                        (1 - alpha_region) * img[y1:y2, x1:x2, c]
                    )

        except Exception as e:
            print(f"한글 텍스트 렌더링 실패: {e}")
            import traceback
            traceback.print_exc()
            # 실패 시 cv2로 시도
            try:
                cv2.putText(img, str(text), pos,
                           cv2.FONT_HERSHEY_SIMPLEX, font_size/30.0, color, 2)
            except:
                pass

    def _get_risk_color(self, risk_value):
        if risk_value >= 0.6:
            return (0, 0, 255)
        elif risk_value >= 0.4:
            return (0, 165, 255)
        return (0, 255, 0)

    def _draw_risk_info(self, frame, mmwave_data, intrusion_count):
        y_offset = 15
        line_height = 16

        cv2.rectangle(frame, (10, 10), (130, 155), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (130, 155), (255, 255, 255), 1)

        cv2.putText(frame, "Risk", (15, y_offset + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        y_offset += 20

        risk_percent = int(self.normalized_risk * 100)
        risk_color = self._get_risk_color(self.normalized_risk)
        cv2.putText(frame, f"Total: {risk_percent}%", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, risk_color, 1)
        y_offset += line_height + 2

        cv2.putText(frame, "Factors:", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        y_offset += line_height - 2

        abbrev = {
            'distance': 'Distance',
            'velocity': 'Velocity',
            'path_intrusion': 'Path',
            'motion_type': 'Motion'
        }

        for name, value in self.risk_factors.items():
            display_name = abbrev.get(name, name[:8])
            cv2.putText(frame, f"{display_name}: {value:.2f}", (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, self._get_risk_color(value), 1)
            y_offset += line_height - 2

    def _time_ago(self, timestamp_str):
        try:
            alert_time = datetime.strptime(timestamp_str, '%H:%M:%S').time()
            now_time = datetime.now().time()
            today = datetime.now().date()
            alert_datetime = datetime.combine(today, alert_time)
            now_datetime = datetime.combine(today, now_time)
            diff = (now_datetime - alert_datetime).total_seconds()
            if diff < 0:
                diff += 86400
            if diff < 60:
                return f"{int(diff)}초 전"
            elif diff < 3600:
                return f"{int(diff // 60)}분 전"
            elif diff < 86400:
                return f"{int(diff // 3600)}시간 전"
            else:
                return f"{int(diff // 86400)}일 전"
        except Exception:
            return timestamp_str

    def _draw_alert_popups(self, frame, right_x_start, right_width):
        popup_height = 120
        start_x = right_x_start
        start_y = 10

        with self.alert_buttons_lock:
            self.alert_close_buttons = []

        with self.alert_lock:
            if len(self.alert_messages) == 0:
                return start_y + popup_height

            alert = self.alert_messages[0]
            alert_type = alert.get('type', 'info')

            if alert_type == 'error' or alert_type == 'danger':
                bg_color = (0, 0, 200)
                border_color = (0, 0, 255)
            elif alert_type == 'warning':
                bg_color = (0, 120, 230)
                border_color = (0, 165, 255)
            elif alert_type == 'success':
                bg_color = (0, 150, 0)
                border_color = (0, 255, 0)
            else:
                bg_color = (50, 50, 50)
                border_color = (180, 180, 180)

            cv2.rectangle(frame, (start_x, start_y),
                          (start_x + right_width, start_y + popup_height),
                          bg_color, -1)

            cv2.line(frame, (start_x, start_y + popup_height),
                     (start_x + right_width, start_y + popup_height),
                     border_color, 3)

            close_btn_size = 25
            close_btn_x = start_x + right_width - close_btn_size - 20
            close_btn_y = start_y + 15

            cv2.circle(frame, (close_btn_x + close_btn_size // 2, close_btn_y + close_btn_size // 2),
                       close_btn_size // 2, (255, 255, 255), -1)

            offset = 6
            cv2.line(frame,
                     (close_btn_x + offset, close_btn_y + offset),
                     (close_btn_x + close_btn_size - offset, close_btn_y + close_btn_size - offset),
                     bg_color, 3)
            cv2.line(frame,
                     (close_btn_x + close_btn_size - offset, close_btn_y + offset),
                     (close_btn_x + offset, close_btn_y + close_btn_size - offset),
                     bg_color, 3)

            with self.alert_buttons_lock:
                self.alert_close_buttons.append((
                    close_btn_x, close_btn_y,
                    close_btn_x + close_btn_size, close_btn_y + close_btn_size,
                    0
                ))

            timestamp = alert.get('timestamp', datetime.now().strftime('%H:%M:%S'))
            time_ago = self._time_ago(timestamp)
            # PIL로 한글 시간 표시
            self._put_korean_text(frame, time_ago, (start_x + 25, start_y + 20),
                                 font_size=22, color=(220, 220, 220))

            message = alert.get('message', 'No message')
            if len(message) > 40:
                message = message[:37] + '...'

            # PIL로 한글 메시지 표시
            self._put_korean_text(frame, message, (start_x + 25, start_y + 60),
                                 font_size=32, color=(255, 255, 255))

        return start_y + popup_height

    def _create_display_layout(self, main_frame):
        """전체 디스플레이 레이아웃 생성 (전체화면 반반 분할)"""
        display_width = self.screen_width
        display_height = self.screen_height

        left_width = display_width // 2
        right_width = display_width - left_width

        display_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)

        # 왼쪽 절반: 카메라 영상 (640x960 → 960x1080에 맞춤)
        cam_h, cam_w = main_frame.shape[:2]  # 960, 640
        scale = display_height / cam_h
        new_cam_w = int(cam_w * scale)
        new_cam_h = display_height
        resized_cam = cv2.resize(main_frame, (new_cam_w, new_cam_h))

        x_offset = (left_width - new_cam_w) // 2
        if x_offset >= 0:
            display_frame[0:new_cam_h, x_offset:x_offset + new_cam_w] = resized_cam
        else:
            crop_start = abs(x_offset)
            display_frame[0:new_cam_h, 0:left_width] = resized_cam[:, crop_start:crop_start + left_width]

        # 오른쪽 절반: 알림 + Team2 패널
        right_x_start = left_width
        display_frame[0:display_height, right_x_start:display_width] = (40, 40, 40)

        popup_height = self._draw_alert_popups(display_frame, right_x_start, right_width)

        # 팝업 밑에 "적재물 낙하 위험" 고정 표시
        warning_y = popup_height + 30
        self._put_korean_text(display_frame, "적재물 낙하 위험",
                             (right_x_start + 40, warning_y),
                             font_size=45, color=(0, 0, 255))

        # ---- Team2 패널 (경고 텍스트 아래)
        panel_x1 = right_x_start
        panel_y1 = warning_y + 70
        panel_x2 = display_width
        panel_y2 = display_height
        self._draw_team2_panel(display_frame, panel_x1, panel_y1, panel_x2, panel_y2)

        # 좌/우 세로 구분선
        cv2.line(display_frame, (left_width, 0), (left_width, display_height), (255, 255, 255), 3)

        return display_frame

    # ---------------- main processing ----------------
    def process_frame(self):
        """프레임 처리 및 표시 (메인 카메라 + 팀2 패널 + 알림)"""
        main_frame = self.get_combined_frame()
        if main_frame is None:
            return None

        if self.mmwave_enabled and self.mmwave_sensor:
            mmwave_data = self.mmwave_sensor.read_detection_status()
        else:
            mmwave_data = {'detected': False, 'distance': None, 'velocity': None, 'connected': False}

        self.intrusion_detected, intrusion_count = self.detect_intrusion(main_frame)
        self.calculate_comprehensive_risk(mmwave_data, intrusion_count)

        main_frame = self.draw_lanes(main_frame)
        self._draw_risk_info(main_frame, mmwave_data, intrusion_count)

        # 위험도 상태 표시 (오른쪽 위)
        if self.normalized_risk >= 0.4:
            if self.normalized_risk >= 0.6:
                text = "STOP"
                color = (0, 0, 255)
            else:
                text = "SLOW"
                color = (0, 165, 255)

            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(main_frame, (640 - text_size[0] - 18, 12),
                          (640 - 8, 35 + text_size[1]), color, -1)
            cv2.putText(main_frame, text, (640 - text_size[0] - 15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        display_frame = self._create_display_layout(main_frame)
        return display_frame

    # ---------------- runtime ----------------
    def start(self):
        print("\n카메라 초기화 중...")
        for i in range(2):
            self.pipelines[i] = self.create_pipeline(i)
            if self.pipelines[i] is None:
                print(f"카메라 {i} 초기화 실패")
                return False

        for i in range(2):
            self.pipelines[i].set_state(Gst.State.PLAYING)

        print("카메라 워밍업 중...")
        time.sleep(2)

        print("\n✓ 듀얼 카메라 시작 완료")
        print("\n키보드 단축키:")
        print("  'q' 또는 ESC - 종료")
        print("  'r' - 배경 모델 리셋")
        print("  'f' - 전체화면 토글")
        print("  't' - 테스트 알림 로드 (test_alert.json)")
        print("\n마우스:")
        print("  알림 팝업 X 버튼 클릭 - 알림 삭제")
        print()

        return True

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            with self.alert_buttons_lock:
                for btn_x1, btn_y1, btn_x2, btn_y2, alert_idx in self.alert_close_buttons:
                    if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
                        with self.alert_lock:
                            if 0 <= alert_idx < len(self.alert_messages):
                                removed_alert = self.alert_messages.pop(alert_idx)
                                print(f"알림 삭제됨: {removed_alert.get('message', 'N/A')}")
                        break

    def _ros_spin_thread(self):
        if not ROS2_AVAILABLE:
            return
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.01)
        except Exception as e:
            self.get_logger().error(f'ROS2 spin 에러: {e}')

    def run(self):
        if not self.start():
            print("카메라 시작 실패")
            return

        window_title = 'Dual Lane Detector + ROS2 Monitor System' if ROS2_AVAILABLE else 'Dual Lane Detector'
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

        if self.use_fullscreen:
            cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print(f"✓ 전체화면 모드 ({self.screen_width}x{self.screen_height})")
        else:
            cv2.resizeWindow(window_title, self.screen_width, self.screen_height)

        cv2.setMouseCallback(window_title, self._mouse_callback)

        if ROS2_AVAILABLE:
            ros_thread = threading.Thread(target=self._ros_spin_thread, daemon=True)
            ros_thread.start()
            print("✓ ROS2 스레드 시작됨")

        try:
            while True:
                frame = self.process_frame()
                if frame is not None:
                    cv2.imshow(window_title, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n종료 중...")
                    break
                elif key == ord('r'):
                    print("배경 모델 리셋")
                    self.bg_subtractors = [
                        cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True),
                        cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
                    ]
                elif key == ord('f'):
                    self.use_fullscreen = not self.use_fullscreen
                    if self.use_fullscreen:
                        cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        print("✓ 전체화면 모드")
                    else:
                        cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        print("✓ 창 모드")
                elif key == ord('t'):
                    try:
                        with open('test_alert.json', 'r', encoding='utf-8') as f:
                            alert_data = json.load(f)
                            alert_data['timestamp'] = datetime.now().strftime('%H:%M:%S')
                            with self.alert_lock:
                                self.alert_messages = [alert_data]
                            print(f"✓ 테스트 알림 로드됨: {alert_data.get('message', '')}")
                    except FileNotFoundError:
                        print("✗ test_alert.json 파일이 없습니다")
                    except json.JSONDecodeError:
                        print("✗ JSON 파싱 실패")

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n\n[Ctrl+C 감지]")
        finally:
            self.stop()

    def stop(self):
        print("\n시스템 종료 중...")

        for i, pipeline in enumerate(self.pipelines):
            if pipeline:
                pipeline.set_state(Gst.State.NULL)
                print(f"  카메라 {i} 정지됨")

        if self.mmwave_enabled and self.mmwave_sensor:
            self.mmwave_sensor.close()
            print("  mmWave 센서 종료됨")

        cv2.destroyAllWindows()
        print("✓ 정리 완료")


def main():
    print("=" * 70)
    if ROS2_AVAILABLE:
        print("  Dual Camera + mmWave + ROS2 Monitor System")
    else:
        print("  Dual Camera + mmWave Detection System")
    print("=" * 70)
    print()

    if ROS2_AVAILABLE:
        rclpy.init()
        print("✓ ROS2 초기화 완료")
    else:
        print("! ROS2 없이 실행 - 카메라 감지 기능만 사용")

    try:
        detector = DualCameraLaneDetector()

        def signal_handler(sig, frame):
            print("\n시그널 수신, 종료 중...")
            detector.stop()
            if ROS2_AVAILABLE:
                rclpy.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        detector.run()

    finally:
        if ROS2_AVAILABLE:
            rclpy.shutdown()


if __name__ == '__main__':
    main()

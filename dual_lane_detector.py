#!/usr/bin/env python3
"""
Dual Camera Lane Intrusion Detection System with mmWave + ROS2
카메라 2대 + mmWave 센서를 통합한 위험도 산정 시스템
ROS2를 통해 원격 서버에서 이미지와 알림 수신
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

        # ROS2 Subscribers (사용 가능한 경우)
        if ROS2_AVAILABLE:
            self.bridge = CvBridge()

            # 두 개의 이미지 토픽 구독
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
        # 카메라 0: 위쪽 (y: 0~480)
        # 카메라 1: 아래쪽 (y: 480~960)
        # 왼쪽 차선: 카메라 0 위쪽에서 시작 → 카메라 1 아래쪽으로 이어짐
        # 오른쪽 차선: 카메라 0 위쪽에서 시작 → 카메라 1 아래쪽으로 이어짐
        self.left_lane = np.array([
            [150, 50],          # 카메라 0 위쪽 왼편
            [50, 910]           # 카메라 1 아래쪽 왼편
        ], dtype=np.int32)

        self.right_lane = np.array([
            [490, 50],          # 카메라 0 위쪽 오른편
            [590, 910]          # 카메라 1 아래쪽 오른편
        ], dtype=np.int32)

        # 배경 차분기 (카메라별)
        self.bg_subtractors = [
            cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True),
            cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        ]

        # GStreamer 초기화
        Gst.init(None)
        self.pipelines = [None, None]

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

            # appsink 가져오기
            appsink = pipeline.get_by_name(f'sink{camera_index}')

            # 콜백 연결 (camera_index를 함께 전달)
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

        # 버퍼에서 데이터 추출
        buffer = sample.get_buffer()
        caps = sample.get_caps()

        # 프레임 크기 가져오기
        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')

        # 버퍼를 numpy 배열로 변환
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        # BGR 이미지로 변환
        frame = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=map_info.data
        )

        # 프레임 복사 및 저장
        with self.frame_locks[camera_index]:
            self.frames[camera_index] = frame.copy()

        buffer.unmap(map_info)
        return Gst.FlowReturn.OK

    def server_image_1_callback(self, msg):
        """ROS2 서버 이미지 1 수신 콜백"""
        if not ROS2_AVAILABLE:
            return

        try:
            # ROS Image를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            with self.server_image_lock:
                self.server_image_1 = cv_image.copy()

        except Exception as e:
            if ROS2_AVAILABLE:
                self.get_logger().error(f'서버 이미지 1 변환 실패: {e}')
            else:
                print(f'서버 이미지 1 변환 실패: {e}')

    def server_image_2_callback(self, msg):
        """ROS2 서버 이미지 2 수신 콜백"""
        if not ROS2_AVAILABLE:
            return

        try:
            # ROS Image를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            with self.server_image_lock:
                self.server_image_2 = cv_image.copy()

        except Exception as e:
            if ROS2_AVAILABLE:
                self.get_logger().error(f'서버 이미지 2 변환 실패: {e}')
            else:
                print(f'서버 이미지 2 변환 실패: {e}')

    def alert_callback(self, msg):
        """ROS2 알림 메시지 수신 콜백"""
        if not ROS2_AVAILABLE:
            return

        try:
            # JSON 파싱
            alert_data = json.loads(msg.data)

            with self.alert_lock:
                # 새 알림이 오면 기존 알림 모두 지우고 새 것만 표시
                self.alert_messages = [alert_data]

            if ROS2_AVAILABLE:
                self.get_logger().info(f'알림 수신: {alert_data}')
            else:
                print(f'알림 수신: {alert_data}')

        except json.JSONDecodeError as e:
            if ROS2_AVAILABLE:
                self.get_logger().error(f'JSON 파싱 실패: {e}')
            else:
                print(f'JSON 파싱 실패: {e}')

    def get_combined_frame(self):
        """2개 카메라 프레임을 세로로 합치기 (위아래)"""
        with self.frame_locks[0]:
            frame0 = self.frames[0].copy() if self.frames[0] is not None else None

        with self.frame_locks[1]:
            frame1 = self.frames[1].copy() if self.frames[1] is not None else None

        # 둘 중 하나라도 없으면 None 반환
        if frame0 is None or frame1 is None:
            return None

        # 세로로 합치기 (위: 카메라 0, 아래: 카메라 1)
        combined = np.vstack([frame0, frame1])
        return combined

    def _calculate_obstacle_presence_risk(self, detected: bool) -> float:
        """장애물 존재 여부에 따른 위험도 (0~1)"""
        return 1.0 if detected else 0.0

    def _calculate_distance_risk(self, distance: float) -> float:
        """거리에 따른 위험도 (0~1)"""
        if distance is None or distance <= 0:
            return 0.0

        if distance <= self.critical_distance:
            return 1.0
        elif distance >= self.safe_distance:
            return 0.0
        else:
            return 1.0 - (distance - self.critical_distance) / (self.safe_distance - self.critical_distance)

    def _calculate_velocity_risk(self, velocity: float) -> float:
        """상대속도에 따른 위험도 (0~1)"""
        if velocity is None:
            return 0.0

        if velocity >= self.safe_velocity:
            return 0.0
        elif velocity <= self.critical_velocity:
            return 1.0
        elif velocity > 0:
            return 0.1
        else:
            return abs(velocity) / abs(self.critical_velocity)

    def _calculate_path_intrusion_risk(self, intrusion_detected: bool, intrusion_count: int) -> float:
        """경로 침범 여부에 따른 위험도 (0~1)"""
        if not intrusion_detected:
            return 0.0

        base_risk = 0.5
        count_factor = min(intrusion_count * 0.2, 0.5)
        return min(base_risk + count_factor, 1.0)

    def _calculate_motion_type_risk(self, is_dynamic: bool, motion_intensity: float = 0.5) -> float:
        """정적/동적 여부에 따른 위험도 (0~1)"""
        if is_dynamic:
            return 0.3 + (motion_intensity * 0.7)
        else:
            return 0.1

    def calculate_comprehensive_risk(self, mmwave_data, intrusion_count):
        """mmWave와 카메라 데이터를 기반으로 통합 위험도 계산"""
        # 센서 연결 여부 확인
        detected = mmwave_data.get('detected', False) and mmwave_data.get('connected', False)

        # 1. 거리
        distance = mmwave_data.get('distance')
        distance_risk = self._calculate_distance_risk(distance) if detected else 0.0

        # 2. 상대속도
        velocity = mmwave_data.get('velocity')
        velocity_risk = self._calculate_velocity_risk(velocity) if detected else 0.0

        # 3. 경로 침범 여부 (카메라)
        path_intrusion_risk = self._calculate_path_intrusion_risk(
            self.intrusion_detected, intrusion_count
        )

        # 4. 정적/동적 여부
        is_dynamic = False
        motion_intensity = 0.0

        if detected and velocity is not None:
            is_dynamic = abs(velocity) > self.motion_velocity_threshold
            if is_dynamic:
                max_velocity = abs(self.critical_velocity)
                motion_intensity = min(abs(velocity) / max_velocity, 1.0)

        motion_type_risk = self._calculate_motion_type_risk(is_dynamic, motion_intensity)

        # 위험 요소 저장
        self.risk_factors = {
            'distance': distance_risk,
            'velocity': velocity_risk,
            'path_intrusion': path_intrusion_risk,
            'motion_type': motion_type_risk
        }

        # 가중 평균
        weights = self.risk_weights
        total_weight = sum(weights.values())

        fused_risk = (
            distance_risk * weights['distance'] +
            velocity_risk * weights['velocity'] +
            path_intrusion_risk * weights['path_intrusion'] +
            motion_type_risk * weights['motion_type']
        ) / total_weight

        # 복합 위험 상황 감지
        high_risk_count = sum(1 for risk in self.risk_factors.values() if risk > 0.7)

        if high_risk_count >= 3:
            fused_risk = min(1.0, fused_risk * 1.3)
        elif high_risk_count >= 2:
            fused_risk = min(1.0, fused_risk * 1.15)

        self.normalized_risk = max(0.0, min(1.0, fused_risk))
        return self.normalized_risk

    def draw_lanes(self, frame):
        """연속된 차선 그리기"""
        # 왼쪽 차선
        cv2.line(frame, tuple(self.left_lane[0]), tuple(self.left_lane[1]),
                 (0, 255, 0), 3)

        # 오른쪽 차선
        cv2.line(frame, tuple(self.right_lane[0]), tuple(self.right_lane[1]),
                 (0, 255, 0), 3)

        # 차선 사이 영역을 ROI로 표시 (반투명 초록색)
        # 시계방향으로 점 정렬: 왼쪽 위 → 오른쪽 위 → 오른쪽 아래 → 왼쪽 아래
        roi_points = np.array([
            self.left_lane[0],   # 왼쪽 위 (150, 50)
            self.right_lane[0],  # 오른쪽 위 (490, 50)
            self.right_lane[1],  # 오른쪽 아래 (590, 910)
            self.left_lane[1]    # 왼쪽 아래 (50, 910)
        ], dtype=np.int32)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_points], (0, 255, 0))
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        # 카메라 경계선 표시 (중앙 가로선)
        cv2.line(frame, (0, 480), (640, 480), (255, 255, 0), 2)

        return frame

    def detect_intrusion(self, frame):
        """차선 침범 감지"""
        if frame is None:
            return False, 0

        # 카메라 0과 1을 분리해서 배경 차분 적용 (위아래로 분리)
        frame0 = frame[:480, :]  # 위쪽 (카메라 0)
        frame1 = frame[480:, :]  # 아래쪽 (카메라 1)

        fg_mask0 = self.bg_subtractors[0].apply(frame0)
        fg_mask1 = self.bg_subtractors[1].apply(frame1)

        # 두 마스크를 세로로 합치기
        fg_mask = np.vstack([fg_mask0, fg_mask1])

        # 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # ROI 마스크 생성 (차선 사이 영역)
        roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        # 시계방향으로 점 정렬
        roi_points = np.array([
            self.left_lane[0],   # 왼쪽 위
            self.right_lane[0],  # 오른쪽 위
            self.right_lane[1],  # 오른쪽 아래
            self.left_lane[1]    # 왼쪽 아래
        ], dtype=np.int32)
        cv2.fillPoly(roi_mask, [roi_points], 255)

        # ROI 영역 내 움직임만 추출
        intrusion_mask = cv2.bitwise_and(fg_mask, roi_mask)

        # 윤곽선 찾기
        contours, _ = cv2.findContours(intrusion_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # 일정 크기 이상의 윤곽선이 있으면 침입으로 간주
        min_area = 1000
        intrusion = False
        intrusion_count = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                intrusion = True
                intrusion_count += 1
                # 침입 영역 표시 (빨간색 윤곽선)
                cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)

                # 침입 영역 중심점 표시
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        return intrusion, intrusion_count

    def process_frame(self):
        """프레임 처리 및 표시 (메인 카메라 + 서버 이미지 + 알림)"""
        # 두 카메라 프레임 합치기
        main_frame = self.get_combined_frame()

        if main_frame is None:
            return None

        # mmWave 센서 데이터 읽기
        mmwave_data = {}
        if self.mmwave_enabled and self.mmwave_sensor:
            mmwave_data = self.mmwave_sensor.read_detection_status()
        else:
            mmwave_data = {
                'detected': False,
                'distance': None,
                'velocity': None,
                'connected': False
            }

        # 침입 감지
        self.intrusion_detected, intrusion_count = self.detect_intrusion(main_frame)

        # 통합 위험도 계산
        self.calculate_comprehensive_risk(mmwave_data, intrusion_count)

        # 차선 그리기
        main_frame = self.draw_lanes(main_frame)

        # 위험도 정보 표시 (왼쪽 위)
        self._draw_risk_info(main_frame, mmwave_data, intrusion_count)

        # 위험도 상태 표시 (오른쪽 위)
        if self.normalized_risk >= 0.4:
            # 위험도 기반: STOP / SLOW
            if self.normalized_risk >= 0.6:
                text = "STOP"
                color = (0, 0, 255)  # 빨간색
            else:  # 0.4 <= risk < 0.6
                text = "SLOW"
                color = (0, 165, 255)  # 주황색

            # 텍스트 배경 (크기 대폭 축소)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(main_frame, (640 - text_size[0] - 18, 12),
                         (640 - 8, 35 + text_size[1]), color, -1)

            # 텍스트 (크기 대폭 축소: 1.0 -> 0.6, 두께: 2 -> 1)
            cv2.putText(main_frame, text, (640 - text_size[0] - 15, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 전체 디스플레이 프레임 생성 (메인 + 서버 정보)
        display_frame = self._create_display_layout(main_frame)

        return display_frame

    def _create_display_layout(self, main_frame):
        """전체 디스플레이 레이아웃 생성 (전체화면 반반 분할)"""
        # 전체화면 크기 (1920x1080)
        display_width = self.screen_width
        display_height = self.screen_height

        # 반반 분할
        left_width = display_width // 2  # 960
        right_width = display_width - left_width  # 960

        # 전체 프레임 생성 (검은 배경)
        display_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)

        # 왼쪽 절반: 카메라 영상 (640x960 → 960x1080에 맞춤)
        # 비율 유지하며 리사이즈
        cam_h, cam_w = main_frame.shape[:2]  # 960, 640

        # 왼쪽 절반에 맞춰 스케일 조정 (세로를 1080에 맞춤)
        scale = display_height / cam_h
        new_cam_w = int(cam_w * scale)
        new_cam_h = display_height

        # 리사이즈
        resized_cam = cv2.resize(main_frame, (new_cam_w, new_cam_h))

        # 왼쪽 중앙 정렬
        x_offset = (left_width - new_cam_w) // 2
        if x_offset >= 0:
            display_frame[0:new_cam_h, x_offset:x_offset+new_cam_w] = resized_cam
        else:
            # 카메라 이미지가 왼쪽 절반보다 크면 크롭
            crop_start = abs(x_offset)
            display_frame[0:new_cam_h, 0:left_width] = resized_cam[:, crop_start:crop_start+left_width]

        # 오른쪽 절반: 알림 팝업 + 서버 이미지 2개 (좌우)
        right_x_start = left_width

        # 오른쪽 패널 배경 (어두운 회색)
        display_frame[0:display_height, right_x_start:display_width] = (40, 40, 40)

        # 알림 팝업 먼저 그리기 (상단)
        popup_height = self._draw_alert_popups(display_frame, right_x_start, right_width)

        # 서버 이미지 영역 (팝업 아래, 화면 절반 높이만)
        img_start_y = popup_height
        img_height = display_height // 2  # 화면의 절반 (540px)
        img_width = right_width // 2  # 각각 480px

        with self.server_image_lock:
            # 서버 이미지 1 (왼쪽)
            if self.server_image_1 is not None:
                img1_resized = cv2.resize(self.server_image_1, (img_width, img_height))
                display_frame[img_start_y:img_start_y+img_height, right_x_start:right_x_start+img_width] = img1_resized
            else:
                cv2.putText(display_frame, "Image 1",
                           (right_x_start + 150, img_start_y + img_height // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 3)

            # 서버 이미지 2 (오른쪽)
            if self.server_image_2 is not None:
                img2_resized = cv2.resize(self.server_image_2, (img_width, img_height))
                display_frame[img_start_y:img_start_y+img_height, right_x_start+img_width:display_width] = img2_resized
            else:
                cv2.putText(display_frame, "Image 2",
                           (right_x_start + img_width + 150, img_start_y + img_height // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 3)

        # 왼쪽/오른쪽 세로 구분선만 (중간 구분선 제거)
        cv2.line(display_frame, (left_width, 0), (left_width, display_height), (255, 255, 255), 3)

        return display_frame

    def _time_ago(self, timestamp_str):
        """시간을 '몇 초 전' 형식으로 변환"""
        try:
            # timestamp 파싱 (HH:MM:SS 형식)
            alert_time = datetime.strptime(timestamp_str, '%H:%M:%S').time()
            now_time = datetime.now().time()

            # 현재 날짜와 결합하여 datetime 객체 생성
            today = datetime.now().date()
            alert_datetime = datetime.combine(today, alert_time)
            now_datetime = datetime.combine(today, now_time)

            # 시간 차이 계산
            diff = (now_datetime - alert_datetime).total_seconds()

            # 음수면 어제 것으로 간주
            if diff < 0:
                diff += 86400  # 24시간 추가

            if diff < 60:
                return f"{int(diff)}초 전"
            elif diff < 3600:
                return f"{int(diff // 60)}분 전"
            elif diff < 86400:
                return f"{int(diff // 3600)}시간 전"
            else:
                return f"{int(diff // 86400)}일 전"
        except:
            return timestamp_str

    def get_alert_popup_height(self):
        """알림 팝업 높이 계산"""
        with self.alert_lock:
            if len(self.alert_messages) == 0:
                return 0
            # 팝업 하나만 표시하므로
            return 100  # 고정 높이

    def _draw_alert_popups(self, frame, right_x_start, right_width):
        """오른쪽 절반 상단에 알림 팝업 (X 버튼 포함)"""
        popup_height = 100
        start_x = right_x_start
        start_y = 0

        # X 버튼 클릭 영역 초기화
        with self.alert_buttons_lock:
            self.alert_close_buttons = []

        with self.alert_lock:
            if len(self.alert_messages) == 0:
                return popup_height

            # 알림 하나만 표시
            alert = self.alert_messages[0]

            # 알림 타입에 따른 색상
            alert_type = alert.get('type', 'info')
            if alert_type == 'error' or alert_type == 'danger':
                bg_color = (0, 0, 200)  # 어두운 빨강
                border_color = (0, 0, 255)  # 빨강
            elif alert_type == 'warning':
                bg_color = (0, 120, 230)  # 어두운 주황
                border_color = (0, 165, 255)  # 주황
            elif alert_type == 'success':
                bg_color = (0, 150, 0)  # 어두운 초록
                border_color = (0, 255, 0)  # 초록
            else:
                bg_color = (50, 50, 50)  # 어두운 회색
                border_color = (180, 180, 180)  # 밝은 회색

            # 배경 (반투명 없이 깔끔하게)
            cv2.rectangle(frame, (start_x, start_y),
                         (start_x + right_width, start_y + popup_height),
                         bg_color, -1)

            # 하단 테두리만 (이쁘게)
            cv2.line(frame, (start_x, start_y + popup_height),
                    (start_x + right_width, start_y + popup_height),
                    border_color, 3)

            # X 버튼 그리기 (오른쪽 위)
            close_btn_size = 25
            close_btn_x = start_x + right_width - close_btn_size - 20
            close_btn_y = start_y + 20

            # X 버튼 배경 (원)
            cv2.circle(frame, (close_btn_x + close_btn_size//2, close_btn_y + close_btn_size//2),
                      close_btn_size//2, (255, 255, 255), -1)

            # X 표시
            offset = 6
            cv2.line(frame,
                    (close_btn_x + offset, close_btn_y + offset),
                    (close_btn_x + close_btn_size - offset, close_btn_y + close_btn_size - offset),
                    bg_color, 3)
            cv2.line(frame,
                    (close_btn_x + close_btn_size - offset, close_btn_y + offset),
                    (close_btn_x + offset, close_btn_y + close_btn_size - offset),
                    bg_color, 3)

            # X 버튼 클릭 영역 저장
            with self.alert_buttons_lock:
                self.alert_close_buttons.append((
                    close_btn_x,
                    close_btn_y,
                    close_btn_x + close_btn_size,
                    close_btn_y + close_btn_size,
                    0
                ))

            # 시간 정보 ("N분 전" 형식)
            timestamp = alert.get('timestamp', '')
            time_ago = self._time_ago(timestamp)
            cv2.putText(frame, time_ago, (start_x + 25, start_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

            # 메시지
            message = alert.get('message', 'No message')
            if len(message) > 60:
                message = message[:57] + '...'

            cv2.putText(frame, message, (start_x + 25, start_y + 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return popup_height

    def _draw_alert_panel_fullscreen(self, frame, start_y, panel_x_start, panel_width):
        """알림 패널 그리기 - 전체화면용 (X 버튼 포함)"""
        panel_x = panel_x_start + 20
        panel_x_end = panel_x_start + panel_width - 20
        y_offset = start_y + 30

        # 제목
        cv2.putText(frame, "Alerts & Notifications", (panel_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        y_offset += 50

        cv2.line(frame, (panel_x, y_offset), (panel_x_end, y_offset), (200, 200, 200), 2)
        y_offset += 30

        # X 버튼 클릭 영역 초기화
        with self.alert_buttons_lock:
            self.alert_close_buttons = []

        with self.alert_lock:
            if len(self.alert_messages) == 0:
                # 알림이 없을 때
                cv2.putText(frame, "No alerts", (panel_x + 300, y_offset + 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            else:
                # 최신 알림부터 표시 (역순)
                actual_index = len(self.alert_messages) - 1
                for i, alert in enumerate(reversed(self.alert_messages)):
                    if y_offset > self.screen_height - 80:  # 화면 밖으로 나가지 않도록
                        break

                    # 알림 타입에 따른 색상
                    alert_type = alert.get('type', 'info')
                    if alert_type == 'error' or alert_type == 'danger':
                        color = (0, 0, 255)  # 빨간색
                    elif alert_type == 'warning':
                        color = (0, 165, 255)  # 주황색
                    elif alert_type == 'success':
                        color = (0, 255, 0)  # 초록색
                    else:
                        color = (255, 255, 255)  # 흰색

                    # 알림 박스
                    box_y1 = y_offset - 5
                    box_y2 = y_offset + 80
                    cv2.rectangle(frame, (panel_x, box_y1),
                                 (panel_x_end, box_y2), color, 3)

                    # X 버튼 그리기 (오른쪽 위)
                    close_btn_size = 25
                    close_btn_x = panel_x_end - close_btn_size - 10
                    close_btn_y = box_y1 + 10

                    # X 버튼 배경 (어두운 회색 원)
                    cv2.circle(frame, (close_btn_x + close_btn_size//2, close_btn_y + close_btn_size//2),
                              close_btn_size//2 + 2, (60, 60, 60), -1)

                    # X 표시 (흰색)
                    offset = 6
                    cv2.line(frame,
                            (close_btn_x + offset, close_btn_y + offset),
                            (close_btn_x + close_btn_size - offset, close_btn_y + close_btn_size - offset),
                            (255, 255, 255), 3)
                    cv2.line(frame,
                            (close_btn_x + close_btn_size - offset, close_btn_y + offset),
                            (close_btn_x + offset, close_btn_y + close_btn_size - offset),
                            (255, 255, 255), 3)

                    # X 버튼 클릭 영역 저장
                    with self.alert_buttons_lock:
                        self.alert_close_buttons.append((
                            close_btn_x,
                            close_btn_y,
                            close_btn_x + close_btn_size,
                            close_btn_y + close_btn_size,
                            actual_index - i  # 실제 리스트 인덱스
                        ))

                    # 시간 정보
                    timestamp = alert.get('timestamp', '')
                    cv2.putText(frame, timestamp, (panel_x + 10, y_offset + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

                    # 메시지
                    message = alert.get('message', 'No message')
                    # 긴 메시지는 자르기 (X 버튼 공간 고려)
                    if len(message) > 50:
                        message = message[:47] + '...'

                    cv2.putText(frame, message, (panel_x + 10, y_offset + 55),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    y_offset += 100

    def _draw_alert_panel(self, frame, start_y):
        """알림 패널 그리기 (X 버튼 포함) - 구버전 (사용 안 함)"""
        panel_x = 650
        y_offset = start_y + 20

        # 제목
        cv2.putText(frame, "Alerts & Notifications", (panel_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30

        cv2.line(frame, (panel_x, y_offset), (1030, y_offset), (200, 200, 200), 1)
        y_offset += 20

        # X 버튼 클릭 영역 초기화
        with self.alert_buttons_lock:
            self.alert_close_buttons = []

        with self.alert_lock:
            if len(self.alert_messages) == 0:
                # 알림이 없을 때
                cv2.putText(frame, "No alerts", (panel_x, y_offset + 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            else:
                # 최신 알림부터 표시 (역순)
                actual_index = len(self.alert_messages) - 1
                for i, alert in enumerate(reversed(self.alert_messages)):
                    if y_offset > 940:  # 화면 밖으로 나가지 않도록
                        break

                    # 알림 타입에 따른 색상
                    alert_type = alert.get('type', 'info')
                    if alert_type == 'error' or alert_type == 'danger':
                        color = (0, 0, 255)  # 빨간색
                    elif alert_type == 'warning':
                        color = (0, 165, 255)  # 주황색
                    elif alert_type == 'success':
                        color = (0, 255, 0)  # 초록색
                    else:
                        color = (255, 255, 255)  # 흰색

                    # 알림 박스
                    box_y1 = y_offset - 5
                    box_y2 = y_offset + 50
                    cv2.rectangle(frame, (panel_x, box_y1),
                                 (1030, box_y2), color, 2)

                    # X 버튼 그리기 (오른쪽 위)
                    close_btn_size = 15
                    close_btn_x = 1030 - close_btn_size - 5
                    close_btn_y = box_y1 + 5

                    # X 버튼 배경 (어두운 회색 원)
                    cv2.circle(frame, (close_btn_x + close_btn_size//2, close_btn_y + close_btn_size//2),
                              close_btn_size//2, (60, 60, 60), -1)

                    # X 표시 (흰색)
                    offset = 4
                    cv2.line(frame,
                            (close_btn_x + offset, close_btn_y + offset),
                            (close_btn_x + close_btn_size - offset, close_btn_y + close_btn_size - offset),
                            (255, 255, 255), 2)
                    cv2.line(frame,
                            (close_btn_x + close_btn_size - offset, close_btn_y + offset),
                            (close_btn_x + offset, close_btn_y + close_btn_size - offset),
                            (255, 255, 255), 2)

                    # X 버튼 클릭 영역 저장
                    with self.alert_buttons_lock:
                        self.alert_close_buttons.append((
                            close_btn_x,
                            close_btn_y,
                            close_btn_x + close_btn_size,
                            close_btn_y + close_btn_size,
                            actual_index - i  # 실제 리스트 인덱스
                        ))

                    # 시간 정보
                    timestamp = alert.get('timestamp', '')
                    cv2.putText(frame, timestamp, (panel_x + 5, y_offset + 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

                    # 메시지
                    message = alert.get('message', 'No message')
                    # 긴 메시지는 자르기 (X 버튼 공간 고려)
                    if len(message) > 35:
                        message = message[:32] + '...'

                    cv2.putText(frame, message, (panel_x + 5, y_offset + 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                    y_offset += 65

    def _draw_risk_info(self, frame, mmwave_data, intrusion_count):
        """위험도 정보 표시 (초컴팩트)"""
        y_offset = 15
        line_height = 16

        # 배경 (너비 절반으로 축소)
        cv2.rectangle(frame, (10, 10), (130, 155), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (130, 155), (255, 255, 255), 1)

        # 제목
        cv2.putText(frame, "Risk", (15, y_offset + 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        y_offset += 20

        # 통합 위험도
        risk_percent = int(self.normalized_risk * 100)
        risk_color = self._get_risk_color(self.normalized_risk)
        cv2.putText(frame, f"Total: {risk_percent}%", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, risk_color, 1)
        y_offset += line_height + 2

        # 위험 요소별 점수 (약어 사용)
        cv2.putText(frame, "Factors:", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        y_offset += line_height - 2

        # 약어 매핑
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

    def _get_risk_color(self, risk_value):
        """위험도에 따른 색상 반환"""
        if risk_value >= 0.6:
            return (0, 0, 255)  # 빨간색 (STOP)
        elif risk_value >= 0.4:
            return (0, 165, 255)  # 주황색 (SLOW)
        else:
            return (0, 255, 0)  # 초록색 (안전)

    def start(self):
        """카메라 시작"""
        print("\n카메라 초기화 중...")

        # 두 카메라 파이프라인 생성
        for i in range(2):
            self.pipelines[i] = self.create_pipeline(i)
            if self.pipelines[i] is None:
                print(f"카메라 {i} 초기화 실패")
                return False

        # 두 카메라 시작
        for i in range(2):
            self.pipelines[i].set_state(Gst.State.PLAYING)

        # 초기화 대기
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
        """마우스 클릭 이벤트 핸들러"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # X 버튼 클릭 확인
            with self.alert_buttons_lock:
                for btn_x1, btn_y1, btn_x2, btn_y2, alert_idx in self.alert_close_buttons:
                    if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
                        # X 버튼 클릭됨 - 해당 알림 삭제
                        with self.alert_lock:
                            if 0 <= alert_idx < len(self.alert_messages):
                                removed_alert = self.alert_messages.pop(alert_idx)
                                print(f"알림 삭제됨: {removed_alert.get('message', 'N/A')}")
                        break

    def run(self):
        """메인 루프"""
        if not self.start():
            print("카메라 시작 실패")
            return

        # 윈도우 생성 및 전체화면 설정
        window_title = 'Dual Lane Detector + ROS2 Monitor' if ROS2_AVAILABLE else 'Dual Lane Detector'
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

        if self.use_fullscreen:
            cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print(f"✓ 전체화면 모드 ({self.screen_width}x{self.screen_height})")
        else:
            cv2.resizeWindow(window_title, self.screen_width, self.screen_height)

        cv2.setMouseCallback(window_title, self._mouse_callback)

        # ROS2 spin을 별도 스레드에서 실행 (ROS2가 사용 가능한 경우)
        if ROS2_AVAILABLE:
            ros_thread = threading.Thread(target=self._ros_spin_thread, daemon=True)
            ros_thread.start()
            print("✓ ROS2 스레드 시작됨")

        try:
            while True:
                # 프레임 처리
                frame = self.process_frame()

                if frame is not None:
                    # 화면 표시
                    cv2.imshow(window_title, frame)

                # 키보드 입력
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # 'q' 또는 ESC
                    print("\n종료 중...")
                    break
                elif key == ord('r'):
                    print("배경 모델 리셋")
                    self.bg_subtractors = [
                        cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True),
                        cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
                    ]
                elif key == ord('f'):
                    # 전체화면 토글
                    self.use_fullscreen = not self.use_fullscreen
                    if self.use_fullscreen:
                        cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        print("✓ 전체화면 모드")
                    else:
                        cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        print("✓ 창 모드")
                elif key == ord('t'):
                    # 테스트 알림 로드 (더미 JSON)
                    try:
                        with open('test_alert.json', 'r', encoding='utf-8') as f:
                            alert_data = json.load(f)
                            # 현재 시간으로 업데이트
                            alert_data['timestamp'] = datetime.now().strftime('%H:%M:%S')
                            with self.alert_lock:
                                self.alert_messages = [alert_data]
                            print(f"✓ 테스트 알림 로드됨: {alert_data['message']}")
                    except FileNotFoundError:
                        print("✗ test_alert.json 파일이 없습니다")
                    except json.JSONDecodeError:
                        print("✗ JSON 파싱 실패")

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n\n[Ctrl+C 감지]")

        finally:
            self.stop()

    def _ros_spin_thread(self):
        """ROS2 spin을 별도 스레드에서 실행"""
        if not ROS2_AVAILABLE:
            return

        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.01)
        except Exception as e:
            if ROS2_AVAILABLE:
                self.get_logger().error(f'ROS2 spin 에러: {e}')
            else:
                print(f'ROS2 spin 에러: {e}')

    def stop(self):
        """시스템 정지"""
        print("\n시스템 종료 중...")

        # 카메라 정지
        for i, pipeline in enumerate(self.pipelines):
            if pipeline:
                pipeline.set_state(Gst.State.NULL)
                print(f"  카메라 {i} 정지됨")

        # mmWave 센서 종료
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

    # ROS2 초기화 (사용 가능한 경우)
    if ROS2_AVAILABLE:
        rclpy.init()
        print("✓ ROS2 초기화 완료")
    else:
        print("! ROS2 없이 실행 - 카메라 감지 기능만 사용")

    try:
        detector = DualCameraLaneDetector()

        # 시그널 핸들러
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
        # ROS2 종료 (사용 가능한 경우)
        if ROS2_AVAILABLE:
            rclpy.shutdown()


if __name__ == '__main__':
    main()

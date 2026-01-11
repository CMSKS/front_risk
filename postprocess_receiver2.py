#!/usr/bin/env python3
import os
import json
import time
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from rclpy.qos import qos_profile_sensor_data


def ensure_dir(p: str):
    """디렉토리 생성 (존재하지 않을 경우)"""
    os.makedirs(p, exist_ok=True)


def now_ms() -> int:
    """현재 시간을 밀리초 단위로 반환"""
    return int(time.time() * 1000)


def status_to_cls(vertical_status: Optional[str]) -> str:
    """vertical_status를 클래스명(danger/safe/error)으로 변환"""
    s = (vertical_status or "").upper()
    if "DANGER" in s:
        return "danger"
    if "SAFE" in s:
        return "safe"
    return "error"


def safe_stem_from_image_filename(img_name: Any) -> Optional[str]:
    """이미지 파일명에서 확장자를 제외한 stem 추출"""
    if isinstance(img_name, str) and len(img_name) > 0:
        base = os.path.basename(img_name)
        stem = os.path.splitext(base)[0]
        return stem if stem else None
    return None


def ext_from_format(fmt: str) -> str:
    """format 문자열에서 파일 확장자 추출"""
    f = (fmt or "").lower()
    if "png" in f:
        return ".png"
    if "jpg" in f or "jpeg" in f:
        return ".jpg"
    if "bmp" in f:
        return ".bmp"
    if "webp" in f:
        return ".webp"
    # 모르면 jpg로 저장(실제 bytes가 png여도 확장자만 틀릴 수 있음)
    return ".jpg"


class Team2JsonReceiver(Node):
    """
    Team2의 분석 결과를 수신하여 저장하는 노드
    
    Subscribes:
        /team2/vertical/result/json (std_msgs/String)
        /team2/vertical/result/image/compressed (sensor_msgs/CompressedImage)
    
    Saves:
        /{danger,safe,error}/...json
        /{danger,safe,error}/...jpg/.png (compressed bytes 그대로 저장)
    """

    def __init__(self):
        super().__init__("team2_output_receiver")

        # ---- JSON 관련 파라미터 ----
        self.declare_parameter("save_root_json", "received_team2_jsons")
        self.declare_parameter("save_json_files", True)
        self.declare_parameter("print_pretty", False)

        # ---- 이미지 관련 파라미터 ----
        self.declare_parameter("save_root_images", "received_team2_images")
        self.declare_parameter("save_image_files", True)
        self.declare_parameter("image_topic", "/team2/vertical/result/image/compressed")
        self.declare_parameter("show_image", False)  # 필요하면 True로
        self.declare_parameter("jpeg_quality_note", False)  # 사용 안 함(확인용)

        # ---- 공통 설정 ----
        self.classes = ["danger", "safe", "error"]
        self.save_root_json = self.get_parameter("save_root_json").get_parameter_value().string_value
        self.save_json_files = bool(self.get_parameter("save_json_files").value)
        self.print_pretty = bool(self.get_parameter("print_pretty").value)

        self.save_root_images = self.get_parameter("save_root_images").get_parameter_value().string_value
        self.save_image_files = bool(self.get_parameter("save_image_files").value)
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.show_image = bool(self.get_parameter("show_image").value)

        # JSON 저장 디렉토리 생성
        if self.save_json_files:
            for c in self.classes:
                ensure_dir(os.path.join(self.save_root_json, c))

        # 이미지 저장 디렉토리 생성
        if self.save_image_files:
            for c in self.classes:
                ensure_dir(os.path.join(self.save_root_images, c))

        # ---- Subscribers ----
        self.sub_json = self.create_subscription(
            String,
            "/team2/vertical/result/json",
            self.cb_json,
            10
        )
        self.sub_img = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.cb_img,
            qos_profile_sensor_data
        )

        # 메시지 카운터
        self.count_json = 0
        self.count_img = 0

        self.get_logger().info(
            "Receiver ready.\n"
            f" JSON topic: /team2/vertical/result/json\n"
            f" IMG topic: {self.image_topic}\n"
            f" save_json_files={self.save_json_files} save_root_json={self.save_root_json}\n"
            f" save_image_files={self.save_image_files} save_root_images={self.save_root_images}\n"
            f" show_image={self.show_image}"
        )

    def cb_json(self, msg: String):
        """JSON 메시지 콜백"""
        self.count_json += 1

        try:
            data: Dict[str, Any] = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"[JSON {self.count_json}] parse failed: {e}")
            return

        v_status = data.get("vertical_status")
        cls = status_to_cls(v_status)

        # image stem 힌트 저장
        stem = safe_stem_from_image_filename(data.get("image_filename"))

        # 로그 출력
        if self.print_pretty:
            pretty = json.dumps(data, ensure_ascii=False, indent=2)
            self.get_logger().info(f"[JSON {self.count_json}] cls={cls}\n{pretty}")
        else:
            self.get_logger().info(
                f"[JSON {self.count_json}] cls={cls} status={v_status} "
                f"dev={data.get('vertical_deviation_deg')} tilt={data.get('vertical_tilt_deg')} "
                f"img={data.get('image_filename')}"
            )

        if not self.save_json_files:
            return

        # 파일명 결정
        out_name = f"{stem}.json" if stem else f"msg_{now_ms()}.json"
        out_path = os.path.join(self.save_root_json, cls, out_name)

        # 중복 파일명 방지
        if os.path.exists(out_path):
            out_path = os.path.join(
                self.save_root_json,
                cls,
                f"{os.path.splitext(out_name)[0]}_{now_ms()}.json"
            )

        # JSON 파일 저장
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.get_logger().warn(f"[JSON {self.count_json}] save failed: {e}")

    def cb_img(self, msg: CompressedImage):
        """압축 이미지 메시지 콜백"""
        self.count_img += 1

        # ✅ frame_id = 파일명 그대로
        out_name = (msg.header.frame_id or "").strip()
        if not out_name:
            out_name = f"img_{now_ms()}.jpg"

        # cls는 항상 "danger"로 고정 (publisher가 danger만 보내니까)
        cls = "danger"
        out_path = os.path.join(self.save_root_images, cls, out_name)

        # 중복 파일 방지
        if os.path.exists(out_path):
            base, ext = os.path.splitext(out_name)
            out_path = os.path.join(
                self.save_root_images,
                cls,
                f"{base}_{now_ms()}{ext}"
            )

        # 저장: compressed bytes 그대로 파일로 write
        if self.save_image_files:
            try:
                with open(out_path, "wb") as f:
                    f.write(bytes(msg.data))
            except Exception as e:
                self.get_logger().warn(f"[IMG {self.count_img}] save failed: {e}")
                return

        self.get_logger().info(
            f"[IMG {self.count_img}] filename={out_name} format={msg.format} "
            f"bytes={len(msg.data)} saved={out_path if self.save_image_files else 'False'}"
        )

        # (옵션) 화면 표시: OpenCV 필요. 설치 안돼있으면 주석/False 유지.
        if self.show_image:
            try:
                import numpy as np
                import cv2

                arr = np.frombuffer(msg.data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    self.get_logger().warn(f"[IMG {self.count_img}] cv2.imdecode failed")
                    return
                cv2.imshow("team2_image", img)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().warn(f"[IMG {self.count_img}] show_image failed: {e}")

    def destroy_node(self):
        """노드 종료 시 정리"""
        # 창 닫기
        try:
            if self.show_image:
                import cv2
                cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = Team2JsonReceiver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
import json
import time
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def now_ms() -> int:
    return int(time.time() * 1000)


def status_to_cls(vertical_status: Optional[str]) -> str:
    s = (vertical_status or "").upper()
    if "DANGER" in s:
        return "danger"
    if "SAFE" in s:
        return "safe"
    # "알수없음", "MISS", 기타 예외는 error로 저장
    return "error"


class Team2JsonReceiver(Node):
    """
    Subscribes:
      /team2/vertical/result/json (std_msgs/String)

    Optionally saves JSON files under:
      <save_root>/{danger,safe,error}/...

    If JSON has:
      - image_filename: use it to name the json file
    Else:
      - fallback to timestamp file name
    """

    def __init__(self):
        super().__init__("team2_output_json_receiver")

        # ---- params
        self.declare_parameter("save_root", "received_team2_jsons")
        self.declare_parameter("save_files", True)
        self.declare_parameter("print_pretty", False)

        self.save_root = self.get_parameter("save_root").get_parameter_value().string_value
        self.save_files = bool(self.get_parameter("save_files").value)
        self.print_pretty = bool(self.get_parameter("print_pretty").value)

        self.classes = ["danger", "safe", "error"]
        if self.save_files:
            for c in self.classes:
                ensure_dir(os.path.join(self.save_root, c))

        # ---- subscriber
        self.sub = self.create_subscription(
            String,
            "/team2/vertical/result/json",
            self.cb,
            10
        )

        self.count = 0
        self.get_logger().info(
            f"Receiver ready.\n"
            f"Subscribing: /team2/vertical/result/json\n"
            f"save_files={self.save_files} save_root={self.save_root}"
        )

    def cb(self, msg: String):
        self.count += 1

        # JSON 파싱
        try:
            data: Dict[str, Any] = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"[{self.count}] JSON parse failed: {e}")
            return

        v_status = data.get("vertical_status")
        cls = status_to_cls(v_status)

        # 로그
        if self.print_pretty:
            pretty = json.dumps(data, ensure_ascii=False, indent=2)
            self.get_logger().info(f"[{self.count}] cls={cls}\n{pretty}")
        else:
            self.get_logger().info(
                f"[{self.count}] cls={cls} status={v_status} "
                f"dev={data.get('vertical_deviation_deg')} tilt={data.get('vertical_tilt_deg')} "
                f"img={data.get('image_filename')}"
            )

        # 저장
        if not self.save_files:
            return

        # 파일명 결정: image_filename이 있으면 그 stem 사용
        img_name = data.get("image_filename")
        if isinstance(img_name, str) and len(img_name) > 0:
            base = os.path.basename(img_name)
            stem = os.path.splitext(base)[0]
            out_name = f"{stem}.json"
        else:
            out_name = f"msg_{now_ms()}.json"

        out_path = os.path.join(self.save_root, cls, out_name)

        # 같은 이름이 이미 있으면 덮어쓰지 않도록 timestamp suffix
        if os.path.exists(out_path):
            out_path = os.path.join(self.save_root, cls, f"{os.path.splitext(out_name)[0]}_{now_ms()}.json")

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.get_logger().warn(f"Save failed: {e}")


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

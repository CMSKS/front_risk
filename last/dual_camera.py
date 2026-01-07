#!/usr/bin/env python3
"""
듀얼 카메라 뷰어 (순수 GStreamer)
두 개의 IMX219 카메라를 동시에 표시합니다
"""

import subprocess
import time
import signal
import sys

class DualCameraViewer:
    def __init__(self):
        self.processes = []

        # 카메라 정보
        self.cameras = [
            {
                'name': 'Camera 0 (i2c@80000)',
                'device': '/base/axi/pcie@120000/rp1/i2c@80000/imx219@10',
            },
            {
                'name': 'Camera 1 (i2c@88000)',
                'device': '/base/axi/pcie@120000/rp1/i2c@88000/imx219@10',
            }
        ]

    def start_camera(self, camera_device, camera_name):
        """단일 카메라 시작"""
        cmd = [
            'gst-launch-1.0',
            'libcamerasrc',
            f'camera-name={camera_device}',
            '!',
            'video/x-raw,width=640,height=480,format=NV21',
            '!',
            'queue',
            '!',
            'videoconvert',
            '!',
            'xvimagesink',
            'sync=false'
        ]

        print(f"[시작] {camera_name}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        self.processes.append({'name': camera_name, 'process': process})
        return process

    def start_all_cameras(self):
        """모든 카메라 시작"""
        print("=" * 70)
        print("  듀얼 카메라 뷰어 (GStreamer)")
        print("=" * 70)
        print()

        for camera in self.cameras:
            self.start_camera(camera['device'], camera['name'])
            time.sleep(0.5)  # 카메라 간 짧은 지연

        print()
        print("=" * 70)
        print("  모든 카메라 실행 중!")
        print("  Ctrl+C를 눌러 종료하세요")
        print("=" * 70)
        print()

    def stop_all(self):
        """모든 카메라 프로세스 종료"""
        print("\n카메라 종료 중...")
        for item in self.processes:
            try:
                item['process'].terminate()
                item['process'].wait(timeout=2)
                print(f"  ✓ {item['name']} 종료됨")
            except subprocess.TimeoutExpired:
                item['process'].kill()
                print(f"  ✓ {item['name']} 강제 종료됨")
        print("모든 카메라 종료 완료.")

    def monitor_cameras(self):
        """카메라 프로세스 모니터링"""
        try:
            while True:
                time.sleep(1)

                # 종료된 프로세스 확인
                for item in self.processes:
                    if item['process'].poll() is not None:
                        print(f"\n⚠ {item['name']} 예기치 않게 종료됨")
                        return False

            return True

        except KeyboardInterrupt:
            print("\n\n[Ctrl+C 감지]")
            return True

    def run(self):
        """메인 실행 함수"""
        self.start_all_cameras()
        self.monitor_cameras()
        self.stop_all()

def main():
    viewer = DualCameraViewer()

    # 시그널 핸들러 설정
    def signal_handler(sig, frame):
        viewer.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    viewer.run()

if __name__ == '__main__':
    main()

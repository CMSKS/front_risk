# front_risk

## 목적

- 대차 전면 위험도 계산 알고리즘 개발을 위함
- 이를 위해, "L0.데이터 입력 → L1.데이터 전처리 → L2.센서 융합 → L3.위험도 계산 → L4.위험도 판별 → L5.출력" 의 전 과정을 설계함

### 설계 내용

1. camera_fusion.py : 라즈베리파이 보드에 연결된 2대의 카메라 융합 후 하나의 파노라마 영상 생성 설계 코드

2. camera.py : 라즈베리파이 보드에 연결된 2대의 카메라 입력 기반으로 융합 후 하나의 파노라마 영상 생성 최종 코드 → L0, L1, L2-1에 대한 설계

   - 카메라 융합 결과
     ![alt text](image/image.png)

3. raw_dump.py : mmWave 로우 데이터 출력 확인 코드

4. sensor_fusion.py : 스티칭된 카메라 영상 + mmWave 로우 데이터 융합 코드

### 최종 코드

1. dual_lane_detector3.py
   - 카메라와 mmWave 센서를 통한 위험도 판단 및 디스플레이 표시하는 코드
2. postprocess_receiver.py
   - ros2 통신을 통한 2팀 랙 기울기 검사 및 위험도 정보를 서버로부터 받아오는 코드

### 디스플레이 이미지 예시

![alt text](<image/Screenshot from 2025-12-28 20-58-18.png>)

- 좌측 : 1팀(대차 정면 영상 및 위험도 판단 결과)
- 우측 : 2팀(파손랙 검사 결과 이미지)

### 진행 상황

1. 2팀 정보를 통신(postprocess_receiver.py)으로 받아 디스플레이에 1팀 정보와 함께 띄우는 것까지 구현(dual_lane_detector3.py)
2. 실행 자동화 구현 필요

   - 첫번째 터미널(ros2실행)

     - pkill -f zenoh
     - cd ~
     - source /opt/ros/jazzy/setup.bash
     - zenoh-bridge-ros2dds -e tcp/192.168.0.12:7447

   - 두번째 터미널(통신 코드 실행)

     - cd ~
     - python3 postprocess_receiver.py

   - 세번째 터미널(디스플레이 실행)
     - cd ~
     - python3 dual_lane_detector.py

### 자동화 스크립트 확인

1. 적용/자동실행 켜기/즉시 실행

   - sudo systemctl daemon-reload
   - sudo systemctl enable demo-zenoh.service demo-postprocess.service demo-dual-lane.service
   - sudo systemctl start demo-zenoh.service demo-postprocess.service demo-dual-lane.service

2. 상태 확인

   - systemctl status demo-zenoh.service
   - systemctl status demo-postprocess.service
   - systemctl status demo-dual-lane.service

3. 로그 실시간 확인

   - journalctl -u demo-zenoh.service -f

4. 재부팅으로 최종 확인

   - sudo reboot

5. 아래 명령이 로그인된 상태에서 동작하는지 확인

   - echo $DISPLAY
   - ls -la ~/.Xauthority

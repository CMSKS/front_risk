#!/usr/bin/env bash
set -e

pkill -f zenoh || true
source /opt/ros/jazzy/setup.bash
exec zenoh-bridge-ros2dds -e tcp/192.168.0.12:7447

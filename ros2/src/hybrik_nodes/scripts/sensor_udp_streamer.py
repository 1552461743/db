#!/usr/bin/env python3
"""Stream suit sensor topics to UDP for external model inference.

This node subscribes to the ROS2 topics published by the half-body ESP32 reader,
then sends the latest sensor state as JSON via UDP at a fixed rate.
"""

from __future__ import annotations

import json
import socket
from typing import Dict, List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField
from std_msgs.msg import Float32, Float32MultiArray


class SensorUdpStreamer(Node):
    def __init__(self) -> None:
        super().__init__("sensor_udp_streamer")

        self.declare_parameter("udp_host", "127.0.0.1")
        self.declare_parameter("udp_port", 5010)
        self.declare_parameter("stream_hz", 30.0)
        self.declare_parameter("imu_channels", 8)

        self.udp_host = str(self.get_parameter("udp_host").value)
        self.udp_port = int(self.get_parameter("udp_port").value)
        self.stream_hz = float(self.get_parameter("stream_hz").value)
        self.imu_channels = int(self.get_parameter("imu_channels").value)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

        self.capacitance: List[float] = []
        self.normalized: List[float] = []
        self.relative_transform: List[float] = []
        self.imu: Dict[str, Dict[str, object]] = {str(i): {} for i in range(self.imu_channels)}

        self.create_subscription(Float32MultiArray, "/sensor/capacitance", self.on_capacitance, 10)
        self.create_subscription(Float32MultiArray, "/sensor/normalized", self.on_normalized, 10)
        self.create_subscription(Float32MultiArray, "/imu/relative_transform", self.on_relative_transform, 10)

        self.imu_subs = []
        self.mag_subs = []
        self.yaw_subs = []
        for channel in range(self.imu_channels):
            self.imu_subs.append(
                self.create_subscription(
                    Imu,
                    f"/imu/channel_{channel}",
                    lambda msg, ch=channel: self.on_imu(ch, msg),
                    10,
                )
            )
            self.mag_subs.append(
                self.create_subscription(
                    MagneticField,
                    f"/imu/channel_{channel}/magnetic_field",
                    lambda msg, ch=channel: self.on_magnetic(ch, msg),
                    10,
                )
            )
            self.yaw_subs.append(
                self.create_subscription(
                    Float32,
                    f"/imu/channel_{channel}/yaw_tilt_deg",
                    lambda msg, ch=channel: self.on_yaw(ch, msg),
                    10,
                )
            )

        self.timer = self.create_timer(1.0 / max(self.stream_hz, 1e-6), self.send_udp)
        self.get_logger().info(
            f"Streaming sensor topics over UDP to {self.udp_host}:{self.udp_port} at {self.stream_hz:.2f} Hz"
        )

    def on_capacitance(self, msg: Float32MultiArray) -> None:
        self.capacitance = [float(v) for v in msg.data]

    def on_normalized(self, msg: Float32MultiArray) -> None:
        self.normalized = [float(v) for v in msg.data]

    def on_relative_transform(self, msg: Float32MultiArray) -> None:
        self.relative_transform = [float(v) for v in msg.data]

    def on_imu(self, channel: int, msg: Imu) -> None:
        self.imu[str(channel)]["linear_acceleration"] = [
            float(msg.linear_acceleration.x),
            float(msg.linear_acceleration.y),
            float(msg.linear_acceleration.z),
        ]
        self.imu[str(channel)]["angular_velocity"] = [
            float(msg.angular_velocity.x),
            float(msg.angular_velocity.y),
            float(msg.angular_velocity.z),
        ]

    def on_magnetic(self, channel: int, msg: MagneticField) -> None:
        self.imu[str(channel)]["magnetic_field"] = [
            float(msg.magnetic_field.x),
            float(msg.magnetic_field.y),
            float(msg.magnetic_field.z),
        ]

    def on_yaw(self, channel: int, msg: Float32) -> None:
        self.imu[str(channel)]["yaw_tilt_deg"] = float(msg.data)

    def send_udp(self) -> None:
        payload = {
            "seq": self.seq,
            "capacitance": self.capacitance,
            "normalized": self.normalized,
            "relative_transform": self.relative_transform,
            "imu": self.imu,
        }
        self.sock.sendto(json.dumps(payload).encode("utf-8"), (self.udp_host, self.udp_port))
        self.seq += 1

    def destroy_node(self):
        try:
            self.sock.close()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = SensorUdpStreamer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # ROS launch may already trigger shutdown on SIGINT.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

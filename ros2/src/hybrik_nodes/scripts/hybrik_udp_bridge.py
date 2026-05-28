#!/usr/bin/env python3
"""Receive HybrIK SMPL predictions over UDP and publish them to a ROS2 topic.

Expected UDP payload:
- JSON object: {"seq": <int>, "data": [float, ...]}
- or plain JSON list: [float, ...]

Published ROS topic type:
- std_msgs/msg/Float32MultiArray
"""

from __future__ import annotations

import json
import socket
from typing import List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class HybrikUdpBridge(Node):
    def __init__(self) -> None:
        super().__init__("hybrik_udp_bridge")

        self.declare_parameter("udp_host", "0.0.0.0")
        self.declare_parameter("udp_port", 5005)
        self.declare_parameter("output_topic", "/hybrik/smpl_24")
        self.declare_parameter("poll_hz", 200.0)

        self.udp_host = str(self.get_parameter("udp_host").value)
        self.udp_port = int(self.get_parameter("udp_port").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.poll_hz = float(self.get_parameter("poll_hz").value)

        self.publisher = self.create_publisher(Float32MultiArray, self.output_topic, 10)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.udp_host, self.udp_port))
        self.socket.setblocking(False)

        self.received_count = 0
        self.timer = self.create_timer(1.0 / self.poll_hz, self.poll_udp)

        self.get_logger().info(
            f"Listening UDP on {self.udp_host}:{self.udp_port}, publishing Float32MultiArray to {self.output_topic}"
        )

    def parse_payload(self, raw_bytes: bytes) -> List[float]:
        payload = json.loads(raw_bytes.decode("utf-8"))

        if isinstance(payload, dict):
            data = payload.get("data")
        else:
            data = payload

        if not isinstance(data, list):
            raise ValueError("UDP payload must be a JSON list or a dict containing a 'data' list")

        return [float(value) for value in data]

    def poll_udp(self) -> None:
        while True:
            try:
                raw_bytes, _ = self.socket.recvfrom(65535)
            except BlockingIOError:
                break
            except Exception as exc:
                self.get_logger().warning(f"UDP receive failed: {exc}")
                break

            try:
                data = self.parse_payload(raw_bytes)
            except Exception as exc:
                self.get_logger().warning(f"Invalid UDP payload: {exc}")
                continue

            message = Float32MultiArray()
            message.data = data
            self.publisher.publish(message)
            self.received_count += 1

    def destroy_node(self):
        try:
            self.socket.close()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = HybrikUdpBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

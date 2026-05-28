#!/usr/bin/env python3
import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class LatestCameraPublisher(Node):
    def __init__(self):
        super().__init__("latest_camera_publisher")

        # ========= 参数 =========
        self.declare_parameter("device", "/dev/video0")
        self.declare_parameter("topic_name", "/dongbu")
        self.declare_parameter("width", 1920)
        self.declare_parameter("height", 1080)
        self.declare_parameter("fps", 30)
        self.declare_parameter("publish_hz", 30.0)
        self.declare_parameter("frame_id", "camera_frame")

        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.topic_name = self.get_parameter("topic_name").get_parameter_value().string_value
        self.width = self.get_parameter("width").get_parameter_value().integer_value
        self.height = self.get_parameter("height").get_parameter_value().integer_value
        self.fps = self.get_parameter("fps").get_parameter_value().integer_value
        self.publish_hz = self.get_parameter("publish_hz").get_parameter_value().double_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value

        # ========= ROS2 =========
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.publisher_ = self.create_publisher(Image, self.topic_name, qos)
        self.bridge = CvBridge()

        # ========= 相机 =========
        self.cap = self._open_camera(self.device)

        # ========= 共享变量 =========
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True

        # ========= FPS统计 =========
        self.frame_count = 0
        self.last_time = time.time()

        # 启动取流线程
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        # 定时发布
        timer_period = 1.0 / self.publish_hz
        self.timer = self.create_timer(timer_period, self._publish_latest_frame)

        self.get_logger().info(
            f"Camera started: {self.device}, topic={self.topic_name}, "
            f"size={self.width}x{self.height}, fps={self.fps}, publish_hz={self.publish_hz}"
        )

    def _open_camera(self, device):
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开相机: {device}")

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        cap.set(cv2.CAP_PROP_FPS, float(self.fps))

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        self.get_logger().info(
            f"实际相机参数: FOURCC={fourcc_str}, "
            f"{actual_width}x{actual_height}, fps={actual_fps:.2f}"
        )

        return cap

    def _capture_loop(self):
        while self.running and rclpy.ok():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.get_logger().warn("读取相机帧失败")
                time.sleep(0.01)
                continue

            # ===== FPS统计 =====
            self.frame_count += 1
            now = time.time()
            if now - self.last_time >= 1.0:
                fps = self.frame_count / (now - self.last_time)
                print(f"[原始相机FPS] {fps:.2f}")
                self.frame_count = 0
                self.last_time = now

            with self.lock:
                self.latest_frame = frame

    def _publish_latest_frame(self):
        frame = None
        with self.lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()

        if frame is None:
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.publisher_.publish(msg)

    def destroy_node(self):
        self.running = False
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)

        if self.cap is not None:
            self.cap.release()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LatestCameraPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

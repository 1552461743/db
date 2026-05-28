#!/usr/bin/env python3
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class ImageViewer(Node):
    def __init__(self):
        super().__init__("image_viewer")

        # 参数
        self.declare_parameter("topic", "/dongbu")
        self.topic = self.get_parameter("topic").get_parameter_value().string_value

        # QoS（和你发布端一致）
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            self.topic,
            self.image_callback,
            qos
        )

        self.get_logger().info(f"订阅话题: {self.topic}")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"转换失败: {e}")
            return
        frame = cv2.resize(frame, (640, 360))   # 👉 改成你想要的大小
        # 显示
        cv2.imshow("dongbu_view", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ImageViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

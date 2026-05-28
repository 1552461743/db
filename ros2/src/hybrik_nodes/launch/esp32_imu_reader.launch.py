from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="hybrik_nodes",
                executable="esp32_imu_reader.py",
                name="esp32_imu_reader",
                output="screen",
                parameters=[
                    {
                        "port": "/dev/ttyACM0",
                        "baudrate": 921600,
                        "timeout": 0.02,
                        "command_interval": 0.05,
                    }
                ],
            ),
        ]
    )

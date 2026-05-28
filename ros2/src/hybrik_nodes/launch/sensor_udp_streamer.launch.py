from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    udp_host = "127.0.0.1"
    udp_port = 5010
    stream_hz = 30.0
    imu_channels = 8

    return LaunchDescription(
        [
            Node(
                package="hybrik_nodes",
                executable="sensor_udp_streamer.py",
                name="sensor_udp_streamer",
                output="screen",
                parameters=[
                    {
                        "udp_host": udp_host,
                        "udp_port": udp_port,
                        "stream_hz": stream_hz,
                        "imu_channels": imu_channels,
                    }
                ],
            ),
        ]
    )

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="hybrik_nodes",
                executable="human_skeleton_tf_publisher",
                name="human_skeleton_tf_publisher",
                output="screen",
                emulate_tty=True,
            ),
        ]
    )

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Manual settings: edit these values directly when switching models.
    repo_root = "/home/nb666/HybrIK"
    checkpoint_path = "/home/nb666/HybrIK/data/yifu_banshen/mod/1/best_model.pt"
    device = "cuda"
    inference_hz = 30.0
    render_hz = 10.0
    output_topic = "/hybrik/smpl_24"
    show_window = True

    return LaunchDescription(
        [
            Node(
                package="hybrik_nodes",
                executable="live_gru_smpl_infer.py",
                name="live_gru_smpl_infer",
                output="screen",
                parameters=[
                    {
                        "repo_root": repo_root,
                        "checkpoint_path": checkpoint_path,
                        "device": device,
                        "inference_hz": inference_hz,
                        "render_hz": render_hz,
                        "output_topic": output_topic,
                        "show_window": show_window,
                    }
                ],
            ),
        ]
    )

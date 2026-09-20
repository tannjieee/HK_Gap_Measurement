"""Launch the installed Orbbec driver with this project's Gemini 335L profile."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration


def generate_launch_description():
    config_path = Path(__file__).resolve().parents[1] / "config" / "gemini335l.yaml"
    upstream = Path(get_package_share_directory("orbbec_camera")) / "launch" / "gemini_330_series.launch.py"
    return LaunchDescription([
        SetEnvironmentVariable(
            "FASTDDS_BUILTIN_TRANSPORTS",
            EnvironmentVariable(
                "FASTDDS_BUILTIN_TRANSPORTS",
                default_value="LARGE_DATA?max_msg_size=8MB&sockets_size=16MB&non_blocking=true",
            ),
        ),
        DeclareLaunchArgument("camera_name", default_value="gemini335l"),
        DeclareLaunchArgument("config_file", default_value=str(config_path)),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(upstream)),
            launch_arguments={
                "camera_name": LaunchConfiguration("camera_name"),
                "config_file_path": LaunchConfiguration("config_file"),
            }.items(),
        ),
    ])

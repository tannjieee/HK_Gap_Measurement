#!/usr/bin/env python3
"""Launch the official Isaac ROS AprilTag component for the HIKROBOT bridge."""

import os
import re

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description() -> LaunchDescription:
    instance = os.environ.get("HIKROBOT_APRILTAG_INSTANCE", "")
    instance = re.sub(r"[^A-Za-z0-9_]", "_", instance).strip("_")
    suffix = f"_{instance}" if instance else ""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "tag_size_m", default_value=os.environ.get("HIKROBOT_APRILTAG_SIZE_M", "0.04835")
            ),
            DeclareLaunchArgument(
                "tag_family", default_value=os.environ.get("HIKROBOT_APRILTAG_FAMILY", "tag36h11")
            ),
            DeclareLaunchArgument(
                "backends", default_value=os.environ.get("HIKROBOT_APRILTAG_BACKENDS", "CUDA")
            ),
            DeclareLaunchArgument(
                "max_tags", default_value=os.environ.get("HIKROBOT_APRILTAG_MAX_TAGS", "64")
            ),
            DeclareLaunchArgument(
                "tile_size", default_value=os.environ.get("HIKROBOT_APRILTAG_TILE_SIZE", "4")
            ),
            ComposableNodeContainer(
                package="rclcpp_components",
                executable="component_container_mt",
                name=f"hikrobot_apriltag_cuda_container{suffix}",
                namespace="",
                output="screen",
                composable_node_descriptions=[
                    ComposableNode(
                        package="isaac_ros_apriltag",
                        plugin="nvidia::isaac_ros::apriltag::AprilTagNode",
                        name=f"hikrobot_apriltag_cuda{suffix}",
                        parameters=[
                            {
                                "size": LaunchConfiguration("tag_size_m"),
                                "tag_family": LaunchConfiguration("tag_family"),
                                "backends": LaunchConfiguration("backends"),
                                "max_tags": LaunchConfiguration("max_tags"),
                                "tile_size": LaunchConfiguration("tile_size"),
                            }
                        ],
                        remappings=[
                            ("image", "image"),
                            ("camera_info", "camera_info"),
                            ("tag_detections", "tag_detections"),
                        ],
                    )
                ],
            ),
        ]
    )


if __name__ == "__main__":
    # Running the file directly lets the GUI start this project-local launch
    # description without pretending that the HK directory is a ROS package.
    from launch import LaunchService

    service = LaunchService()
    service.include_launch_description(generate_launch_description())
    raise SystemExit(service.run())

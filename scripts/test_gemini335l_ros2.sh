#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/scripts/gemini335l_ros_env.sh"
# Use an isolated DDS domain for an owned test launch; the normal launch script
# retains ROS_DOMAIN_ID so it can communicate with the user's other ROS nodes.
exec /usr/bin/python3 "$project_root/scripts/check_gemini335l_topics.py" \
    --start-driver --domain-id "${GEMINI_TEST_DOMAIN_ID:-221}" "$@"

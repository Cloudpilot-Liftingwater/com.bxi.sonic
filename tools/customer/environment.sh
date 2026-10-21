#!/usr/bin/env bash
# Source explicitly; sets paths only, never starts services/cameras/robot.
_sonic_customer_env() {
  local repo config had_u=0 setup_file
  repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd) || return
  config=${SONIC_SITE_CONFIG:?Set SONIC_SITE_CONFIG to your reviewed site JSON}
  export SONIC_SITE_CONFIG="$config"
  local assignments
  assignments=$(/usr/bin/python3 -B "$repo/tools/customer/site_environment.py") || return
  eval "$assignments"  # all values are shell-quoted by shlex.quote
  [[ $- == *u* ]] && had_u=1 && set +u
  unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
  for setup_file in "$SONIC_ROS_SETUP" "$SONIC_CONTROLLER_SETUP"; do
    if ! source "$setup_file"; then
      [[ $had_u == 1 ]] && set -u
      return 1
    fi
  done
  [[ $had_u == 1 ]] && set -u
  export ROS_DOMAIN_ID="$SONIC_SITE_DOMAIN" PYTHONDONTWRITEBYTECODE=1 COLLECTION_PROFILE=real
  export BXI_LOCAL_ROS_PREFIX="$repo/tools/customer/runtime/ros_humble_x86_64"
  export AMENT_PREFIX_PATH="$BXI_LOCAL_ROS_PREFIX${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
  export LD_LIBRARY_PATH="$BXI_LOCAL_ROS_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PYTHONPATH="$repo/tools/pico_collection/sonic_pico_collection_20261019_r7/collector${PYTHONPATH:+:$PYTHONPATH}"
  export SONIC_XRT_SERVICE_DIR="$repo/runtime/linux-x86_64/roboticsservice"
}
_sonic_customer_env
_sonic_env_result=$?
unset -f _sonic_customer_env
return "$_sonic_env_result"

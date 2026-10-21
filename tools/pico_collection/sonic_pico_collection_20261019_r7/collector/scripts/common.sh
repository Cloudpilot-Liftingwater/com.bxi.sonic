#!/usr/bin/env bash

# Shared environment/config helpers for all operator-facing shell entrypoints.
# This file is sourced; callers are expected to enable their own shell options.

COLLECTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLLECTION_WORKSPACE_ROOT="$(cd "${COLLECTION_DIR}/.." && pwd)"
COLLECTION_CONTROLLER_ROOT="${BXI_CONTROLLER_ROOT:-${COLLECTION_WORKSPACE_ROOT}/bxi_rl_controller_ros2_example}"
COLLECTION_CONTROLLER_PACKAGE="bxi_example_py_elf3"
COLLECTION_LOCAL_ROS_PREFIX="${BXI_LOCAL_ROS_PREFIX:-${COLLECTION_DIR}/.ros_local_root/opt/ros/humble}"

collection_prepend_path() {
  local variable_name="$1"
  local value="$2"
  local current="${!variable_name:-}"
  if [[ -z "${current}" ]]; then
    printf -v "${variable_name}" '%s' "${value}"
  elif [[ ":${current}:" != *":${value}:"* ]]; then
    printf -v "${variable_name}" '%s:%s' "${value}" "${current}"
  fi
  export "${variable_name}"
}

collection_configure_local_ros_prefix() {
  [[ -d "${COLLECTION_LOCAL_ROS_PREFIX}/share/ament_index" ]] || return 0
  collection_prepend_path AMENT_PREFIX_PATH "${COLLECTION_LOCAL_ROS_PREFIX}"
  collection_prepend_path CMAKE_PREFIX_PATH "${COLLECTION_LOCAL_ROS_PREFIX}"
  collection_prepend_path LD_LIBRARY_PATH "${COLLECTION_LOCAL_ROS_PREFIX}/lib/x86_64-linux-gnu"
  collection_prepend_path LD_LIBRARY_PATH "${COLLECTION_LOCAL_ROS_PREFIX}/lib"
  collection_prepend_path PATH "${COLLECTION_LOCAL_ROS_PREFIX}/bin"
  if [[ -d "${COLLECTION_LOCAL_ROS_PREFIX}/lib/python3.10/site-packages" ]]; then
    collection_prepend_path PYTHONPATH \
      "${COLLECTION_LOCAL_ROS_PREFIX}/lib/python3.10/site-packages"
  fi
}

collection_source_setup_safely() {
  local setup_file="$1"
  local had_nounset=0
  [[ -f "${setup_file}" ]] || return 0
  if [[ $- == *u* ]]; then
    had_nounset=1
    set +u
  fi
  # shellcheck disable=SC1090
  source "${setup_file}"
  if [[ "${had_nounset}" -eq 1 ]]; then
    set -u
  fi
}

collection_install_contains_package() {
  local setup_file="$1"
  local install_prefix="${setup_file%/setup.bash}"
  local package="$2"
  [[ -f "${install_prefix}/share/ament_index/resource_index/packages/${package}" \
    || -f "${install_prefix}/${package}/share/ament_index/resource_index/packages/${package}" ]]
}

collection_find_controller_setup() {
  local candidate
  local -a candidates=(
    "${COLLECTION_CONTROLLER_ROOT}/install/setup.bash"
    "${HOME}/bxi_ws/bxi_rl_controller_ros2_example/install/setup.bash"
    "${HOME}/install/setup.bash"
  )

  if [[ -n "${BXI_INSTALL_SETUP:-}" ]]; then
    [[ -f "${BXI_INSTALL_SETUP}" ]] || {
      echo "BXI_INSTALL_SETUP does not exist: ${BXI_INSTALL_SETUP}" >&2
      return 2
    }
    printf '%s\n' "${BXI_INSTALL_SETUP}"
    return 0
  fi

  for candidate in "${candidates[@]}"; do
    if collection_install_contains_package \
      "${candidate}" "${COLLECTION_CONTROLLER_PACKAGE}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

collection_find_system_ros_domain_id() {
  local unit_file="/etc/systemd/system/ros_elf_launch.service"
  local rc_environment="/opt/bxi/bxi_rc_ros2/env.conf"
  local value=""

  # The factory controller launcher is authoritative on deployed robots.  It
  # may intentionally differ from the newer gateway default in env.conf.
  if [[ -r "${unit_file}" ]]; then
    value="$(sed -nE \
      's/.*export[[:space:]]+ROS_DOMAIN_ID=([0-9]+).*/\1/p' \
      "${unit_file}" | head -n 1)"
  fi
  if [[ -z "${value}" && -r "${rc_environment}" ]]; then
    value="$(sed -nE \
      's/^[[:space:]]*ROS_DOMAIN_ID[[:space:]]*=[[:space:]]*([0-9]+).*/\1/p' \
      "${rc_environment}" | head -n 1)"
  fi
  [[ -z "${value}" ]] || printf '%s\n' "${value}"
}

collection_configure_ros_domain() {
  local profile="${1:-}"
  local detected=""
  if [[ -n "${BXI_ROS_DOMAIN_ID:-}" ]]; then
    export ROS_DOMAIN_ID="${BXI_ROS_DOMAIN_ID}"
    return 0
  fi
  if [[ "${profile}" != "real" ]]; then
    # Never inherit the factory hardware Domain for simulation. Both
    # controllers subscribe to the global /motion_commands topic. Domain 42 is
    # reserved here for the VLA sim/hybrid stack so a factory controller on
    # Domain 0 can neither be mistaken for MuJoCo nor receive sim commands.
    export ROS_DOMAIN_ID="${BXI_SIM_ROS_DOMAIN_ID:-42}"
    return 0
  fi
  if [[ -n "${ROS_DOMAIN_ID:-}" ]]; then
    return 0
  fi
  detected="$(collection_find_system_ros_domain_id)"
  if [[ -n "${detected}" ]]; then
    export ROS_DOMAIN_ID="${detected}"
  fi
}

collection_configure_fastdds_transport() {
  local profile="${1:-}"
  [[ "${profile}" == "real" ]] || return 0
  # Factory robot processes run as root while collection normally runs as the
  # operator user. Fast DDS may discover across that boundary but then select
  # a root-owned shared-memory segment that the subscriber cannot open. Force
  # the interoperable localhost UDP path unless the operator supplied an
  # explicit Fast DDS transport policy.
  if [[ -z "${FASTDDS_BUILTIN_TRANSPORTS:-}" ]]; then
    export FASTDDS_BUILTIN_TRANSPORTS="${BXI_FASTDDS_TRANSPORTS:-UDPv4}"
  fi
}

collection_source_environment() {
  local profile="${1:-${COLLECTION_PROFILE:-}}"
  local controller_setup=""
  collection_source_setup_safely /opt/ros/humble/setup.bash
  collection_source_setup_safely /opt/bxi/bxi_ros2_pkg/setup.bash
  if controller_setup="$(collection_find_controller_setup)"; then
    collection_source_setup_safely "${controller_setup}"
    export BXI_INSTALL_SETUP_RESOLVED="${controller_setup}"
  fi
  collection_configure_local_ros_prefix
  collection_configure_ros_domain "${profile}"
  collection_configure_fastdds_transport "${profile}"
  export PYTHONPATH="${COLLECTION_DIR}:${PYTHONPATH:-}"
}

collection_require_profile() {
  case "$1" in
    sim|hybrid_debug|real) ;;
    *)
      echo "Mode must be sim, hybrid_debug, or real: $1" >&2
      return 2
      ;;
  esac
}

collection_default_config() {
  case "$1" in
    sim) printf '%s\n' "${COLLECTION_DIR}/config/site_sim.yaml" ;;
    hybrid_debug) printf '%s\n' "${COLLECTION_DIR}/config/site_hybrid.yaml" ;;
    real) printf '%s\n' "${COLLECTION_DIR}/config/site_real.yaml" ;;
    *) return 2 ;;
  esac
}

collection_absolute_path() {
  local path="$1"
  if [[ "${path}" == /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${COLLECTION_DIR}/${path#./}"
  fi
}

collection_validate_config() {
  local profile="$1"
  local config="$2"
  local configured_mode

  collection_require_profile "${profile}"
  [[ -f "${config}" ]] || {
    echo "Configuration file not found: ${config}" >&2
    return 2
  }
  python3 "${COLLECTION_DIR}/check_config.py" "${config}"
  if [[ "${profile}" == "real" ]]; then
    python3 "${COLLECTION_DIR}/check_config.py" "${config}" --formal
  fi
  configured_mode="$(python3 -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["mode"])' "${config}")"
  if [[ "${configured_mode}" != "${profile}" ]]; then
    echo "Mode mismatch: command selected ${profile}, but ${config} declares ${configured_mode}" >&2
    return 2
  fi
}

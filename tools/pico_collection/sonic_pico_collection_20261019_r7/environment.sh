#!/usr/bin/env bash
# Customer-capable paths; the robot's installed frozen r7 is not modified.
_sonic_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd) || return
source "$_sonic_repo/tools/customer/environment.sh"

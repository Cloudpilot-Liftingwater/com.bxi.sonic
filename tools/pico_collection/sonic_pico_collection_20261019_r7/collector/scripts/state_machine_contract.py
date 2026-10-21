#!/usr/bin/env python3
"""Parse and validate the controller state-machine info ROS message."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import yaml


SONIC_STATE = "com.bxi.sonic/sonic_teleop"
SONIC_ACTIVATE_EVENT = "com.bxi.sonic/activate"
SONIC_ACTIVATE_SLOT = "btn_10"
SONIC_ACTIVATE_VALUE = 9


class ContractError(ValueError):
    """The state topic is readable, but it is not this workspace's contract."""


def decode_ros_echo(text: str) -> dict[str, Any]:
    document = yaml.safe_load(text.split("\n---", 1)[0])
    if not isinstance(document, dict) or "data" not in document:
        raise ContractError("state topic did not contain a std_msgs/String data field")
    try:
        payload = json.loads(document["data"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ContractError("state topic data was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ContractError("state topic JSON was not an object")
    return payload


def current_state(payload: dict[str, Any]) -> str:
    try:
        value = payload["current"]["name"]
    except (KeyError, TypeError) as exc:
        raise ContractError("state topic did not contain current.name") from exc
    if not isinstance(value, str) or not value:
        raise ContractError("state topic current.name was empty")
    return value


def validate_sonic_controller(payload: dict[str, Any]) -> dict[str, Any]:
    current = current_state(payload)
    graph = payload.get("graph")
    if not isinstance(graph, dict):
        raise ContractError("state topic did not contain graph metadata")

    states = graph.get("states")
    state_names = (
        {
            item.get("name")
            for item in states
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        if isinstance(states, list)
        else set()
    )
    if SONIC_STATE not in state_names:
        legacy_hint = (
            " (legacy short-name controller detected)"
            if current
            in {"normal", "recover", "pd_brake", "zero_torque", "sonic_teleop"}
            else ""
        )
        raise ContractError(
            f"missing state {SONIC_STATE!r}; current={current!r}{legacy_hint}"
        )

    remote_events = graph.get("remote_events")
    event = (
        remote_events.get(SONIC_ACTIVATE_EVENT)
        if isinstance(remote_events, dict)
        else None
    )
    if not isinstance(event, dict):
        raise ContractError(f"missing remote event {SONIC_ACTIVATE_EVENT!r}")
    if (
        event.get("slot") != SONIC_ACTIVATE_SLOT
        or event.get("value") != SONIC_ACTIVATE_VALUE
    ):
        raise ContractError(
            "unexpected Sonic activation mapping: "
            f"expected {SONIC_ACTIVATE_SLOT}={SONIC_ACTIVATE_VALUE}, got "
            f"{event.get('slot')}={event.get('value')}"
        )

    return {
        "compatible": True,
        "current": current,
        "sonic_state": SONIC_STATE,
        "activate": f"{SONIC_ACTIVATE_SLOT}={SONIC_ACTIVATE_VALUE}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("current", "validate"))
    args = parser.parse_args()
    try:
        payload = decode_ros_echo(sys.stdin.read())
        if args.action == "current":
            print(current_state(payload))
        else:
            print(json.dumps(validate_sonic_controller(payload), ensure_ascii=False))
    except ContractError as exc:
        print(f"incompatible controller: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

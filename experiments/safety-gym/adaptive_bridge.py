#!/usr/bin/env python3
"""Feature-profile guards and deterministic adaptive-run manifest utilities.

This module does not read credentials or perform model requests. The Gym agent
owns API traffic; this layer pins inputs and refuses incompatible resumes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "adaptive-runtime-v1"
STAGES = (
    (0, 12_800),
    (12_800, 38_400),
    (38_400, 89_600),
    (89_600, None),
)
ASR_LENGTHS = (
    (1.0, 0, 0, 0, 0, 0, 0),
    (0.50, 0.25, 0.25, 0, 0, 0, 0),
    (0.30, 0.15, 0.20, 0.20, 0.15, 0, 0),
    (0.20, 0.10, 0.10, 0.15, 0.15, 0.15, 0.15),
)
OR_LENGTHS = (
    (1.0, 0, 0, 0, 0, 0, 0),
    (0.50, 0.25, 0.25, 0, 0, 0, 0),
    (0.40, 0.30, 0.30, 0, 0, 0, 0),
    (0.40, 0.30, 0.30, 0, 0, 0, 0),
)


def profile_for(*, adaptive: bool, honeypot: bool, bridge_control: bool = False) -> str:
    """Map the four supported feature switch combinations to a run profile."""
    if bridge_control and (not adaptive or honeypot):
        raise ValueError("bridge_control requires adaptive=true and honeypot=false")
    if not adaptive and honeypot:
        return "honeypot_single_turn"
    if adaptive and honeypot:
        return "adaptive_honeypot"
    if bridge_control:
        return "single_turn_bridge"
    if adaptive:
        return "adaptive"
    return "legacy"


def validate_profile(profile: str, *, adaptive: bool, honeypot: bool,
                     bridge_control: bool = False) -> None:
    expected = profile_for(adaptive=adaptive, honeypot=honeypot,
                           bridge_control=bridge_control)
    if profile != expected:
        raise ValueError(f"profile {profile!r} conflicts with switches; expected {expected!r}")
    if profile == "legacy":
        return
    if profile not in {"adaptive", "single_turn_bridge", "honeypot_single_turn",
                       "adaptive_honeypot"}:
        raise ValueError(f"unknown profile: {profile}")
    if profile == "adaptive" and not adaptive:
        raise ValueError("adaptive profile requires adaptive mode")
    if profile in {"honeypot_single_turn", "adaptive_honeypot"} and not honeypot:
        raise ValueError("honeypot profile requires the honeypot switch")


def validate_aliases(aliases: Mapping[str, str]) -> None:
    required = {"defender", "attacker", "judge", "validator"}
    if set(aliases) != required or any(not isinstance(v, str) or not v.strip()
                                       for v in aliases.values()):
        raise ValueError("model aliases must define defender, attacker, judge and validator")
    if aliases["defender"] != "policy_model":
        raise ValueError("defender must route through the native trainable policy_model")
    if aliases["attacker"] == aliases["defender"]:
        raise ValueError("attacker alias must be separate from the policy model")
    if aliases["validator"] == aliases["defender"]:
        raise ValueError("validator alias must be separate from the policy model")
    if aliases["judge"] == aliases["defender"]:
        raise ValueError("judge alias must be separate from the policy model")


def curriculum_stage(*, ordinal: int | None, ordinal_attested: bool,
                     immutable_stage_id: int | None = None) -> int:
    """Select a stage only from collector-attested ordinals or a frozen pilot id."""
    if ordinal is not None:
        if not ordinal_attested:
            raise ValueError("native prompt ordinal is present without collector attestation")
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("native prompt ordinal must be a nonnegative integer")
        for index, (start, end) in enumerate(STAGES):
            if ordinal >= start and (end is None or ordinal < end):
                return index
        raise AssertionError("unreachable ordinal stage")
    if immutable_stage_id is None:
        raise ValueError("curriculum requires an attested ordinal or immutable stage_id")
    if type(immutable_stage_id) is not int or not 0 <= immutable_stage_id < len(STAGES):
        raise ValueError("immutable stage_id must be 0..3")
    return immutable_stage_id


def planned_length(*, mode: str, stage: int, seed: int, source_id: str,
                   ordinal: int | None, stage_id: int | None,
                   rollout_index: int, max_turns: int = 7) -> int:
    """Draw one deterministic group-shared cap from the frozen stage distribution."""
    if mode not in {"asr", "or"}:
        raise ValueError("mode must be asr or or")
    if type(stage) is not int or not 0 <= stage < 4:
        raise ValueError("stage must be 0..3")
    if not source_id:
        raise ValueError("source_id is required for stable horizon sampling")
    if not 1 <= max_turns <= 10:
        raise ValueError("absolute configured turn ceiling must be 1..10")
    if mode == "or":
        max_turns = min(max_turns, 3)
    distribution = ASR_LENGTHS[stage] if mode == "asr" else OR_LENGTHS[stage]
    available = min(max_turns, len(distribution))
    weights = distribution[:available]
    total = sum(weights)
    if total <= 0:
        raise ValueError("stage distribution has no probability under the configured cap")
    key = {
        "seed": seed, "source_id": source_id, "ordinal": ordinal,
        "stage_id": stage_id, "stage": stage, "mode": mode,
    }
    digest = hashlib.sha256(canonical_json(key)).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64 * total
    cumulative = 0.0
    for length, weight in enumerate(weights, start=1):
        cumulative += weight
        if draw < cumulative:
            return length
    return available


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(inputs: Mapping[str, Path], pinned: Mapping[str, Any]) -> dict[str, Any]:
    if not inputs or not pinned:
        raise ValueError("manifest requires input files and pinned run metadata")
    file_hashes = {}
    for name, path in sorted(inputs.items()):
        if not path.is_file():
            raise FileNotFoundError(f"manifest input is missing: {name}")
        file_hashes[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "inputs": file_hashes,
        "pinned": dict(pinned),
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    return payload


def write_manifest(path: Path, manifest: Mapping[str, Any], *, resume: bool = False) -> None:
    """A resume is allowed only when the byte-stable objective manifest matches."""
    encoded = canonical_json(manifest) + b"\n"
    if path.exists():
        if not resume:
            raise FileExistsError(f"run manifest already exists: {path}")
        existing = path.read_bytes()
        if existing != encoded:
            raise ValueError("adaptive objective/data/runtime changed; start a fresh run id")
        return
    if resume:
        raise FileNotFoundError(f"cannot resume without the original run manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def _parse_manifest_inputs(values: Sequence[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("input must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        if not name or name in result:
            raise ValueError("manifest input names must be unique and nonempty")
        result[name] = Path(raw_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--profile", required=True)
    profile_parser.add_argument("--adaptive", action="store_true")
    profile_parser.add_argument("--honeypot", action="store_true")
    profile_parser.add_argument("--bridge-control", action="store_true")
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("--resume", action="store_true")
    manifest_parser.add_argument("--input", action="append", default=[])
    manifest_parser.add_argument("--pinned-json", required=True)
    args = parser.parse_args()
    if args.command == "profile":
        validate_profile(args.profile, adaptive=args.adaptive, honeypot=args.honeypot,
                         bridge_control=args.bridge_control)
        return
    inputs = _parse_manifest_inputs(args.input)
    pinned = json.loads(args.pinned_json)
    manifest = build_manifest(inputs, pinned)
    write_manifest(args.output, manifest, resume=args.resume)


if __name__ == "__main__":
    main()

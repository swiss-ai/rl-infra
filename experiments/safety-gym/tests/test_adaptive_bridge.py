from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adaptive_bridge as bridge


def test_four_modes_and_single_turn_control():
    assert bridge.profile_for(adaptive=False, honeypot=False) == "legacy"
    assert bridge.profile_for(adaptive=True, honeypot=False) == "adaptive"
    assert bridge.profile_for(adaptive=False, honeypot=True) == "honeypot_single_turn"
    assert bridge.profile_for(adaptive=True, honeypot=True) == "adaptive_honeypot"
    assert bridge.profile_for(adaptive=True, honeypot=False, bridge_control=True) == "single_turn_bridge"
    with pytest.raises(ValueError):
        bridge.profile_for(adaptive=False, honeypot=True, bridge_control=True)


def test_profile_conflicts_fail_closed():
    with pytest.raises(ValueError):
        bridge.validate_profile("adaptive", adaptive=False, honeypot=False)
    with pytest.raises(ValueError):
        bridge.validate_profile("adaptive_honeypot", adaptive=True, honeypot=False)


def test_aliases_keep_policy_route_separate():
    valid = {"defender": "policy_model", "attacker": "adaptive_attacker_model",
             "judge": "adaptive_judge_model", "validator": "adaptive_validator_model"}
    bridge.validate_aliases(valid)
    invalid = dict(valid, attacker="policy_model")
    with pytest.raises(ValueError):
        bridge.validate_aliases(invalid)


@pytest.mark.parametrize(("ordinal", "stage"), [
    (0, 0), (12_799, 0), (12_800, 1), (38_399, 1),
    (38_400, 2), (89_599, 2), (89_600, 3), (3_000_000, 3),
])
def test_attested_ordinal_stage_boundaries(ordinal, stage):
    assert bridge.curriculum_stage(ordinal=ordinal, ordinal_attested=True) == stage


def test_unattested_or_hashed_task_index_cannot_drive_stage():
    with pytest.raises(ValueError, match="attestation"):
        bridge.curriculum_stage(ordinal=(1 << 62), ordinal_attested=False)
    with pytest.raises(ValueError, match="immutable"):
        bridge.curriculum_stage(ordinal=None, ordinal_attested=False)


def test_immutable_pilot_stage_is_supported():
    assert bridge.curriculum_stage(ordinal=None, ordinal_attested=False,
                                   immutable_stage_id=2) == 2


def test_horizon_is_deterministic_and_shared_by_group():
    args = dict(mode="asr", stage=3, seed=17_017, source_id="task-1",
                ordinal=55, stage_id=None, max_turns=7)
    first = bridge.planned_length(**args, rollout_index=0)
    assert bridge.planned_length(**args, rollout_index=15) == first
    assert 1 <= first <= 7


def test_or_ceiling_is_three_and_absolute_ceiling_is_ten():
    args = dict(mode="or", stage=3, seed=1, source_id="task-2",
                ordinal=100_000, stage_id=None, rollout_index=0, max_turns=10)
    assert bridge.planned_length(**args) <= 3
    args["max_turns"] = 11
    with pytest.raises(ValueError):
        bridge.planned_length(**args)


def test_manifest_resume_pins_objective_and_files(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text("id:one\n")
    manifest = bridge.build_manifest({"train": data}, {"objective": "adaptive-v1"})
    path = tmp_path / "run" / "manifest.json"
    bridge.write_manifest(path, manifest)
    bridge.write_manifest(path, manifest, resume=True)
    changed = bridge.build_manifest({"train": data}, {"objective": "adaptive-v2"})
    with pytest.raises(ValueError, match="fresh run"):
        bridge.write_manifest(path, changed, resume=True)

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from pathlib import Path

import torch


_G1_SYMMETRY_PATH = (
    Path(__file__).resolve().parents[1] / "legged_lab" / "tasks" / "locomotion" / "amp" / "mdp" / "symmetry" / "g1.py"
)
_SPEC = importlib.util.spec_from_file_location("g1_symmetry", _G1_SYMMETRY_PATH)
g1_symmetry = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(g1_symmetry)
_flip_height_scan_left_right = g1_symmetry._flip_height_scan_left_right


def _fake_env(ordering: str):
    sensor = SimpleNamespace(cfg=SimpleNamespace(shape=(2, 3), pattern_cfg=SimpleNamespace(ordering=ordering)))
    return SimpleNamespace(scene=SimpleNamespace(sensors={"height_scanner": sensor}))


def test_flip_height_scan_left_right_preserves_xy_flat_ordering() -> None:
    height_scan = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])

    flipped = _flip_height_scan_left_right(_fake_env("xy"), height_scan)

    assert torch.equal(flipped, torch.tensor([[4.0, 5.0, 2.0, 3.0, 0.0, 1.0]]))


def test_flip_height_scan_left_right_preserves_yx_flat_ordering() -> None:
    height_scan = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])

    flipped = _flip_height_scan_left_right(_fake_env("yx"), height_scan)

    assert torch.equal(flipped, torch.tensor([[2.0, 1.0, 0.0, 5.0, 4.0, 3.0]]))

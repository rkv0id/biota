"""Tests for the v1.1.0 descriptor registration system.

Covers:
- The Descriptor dataclass and REGISTRY contents
- resolve_descriptors() validation
- Each new built-in descriptor (oscillation, compactness, mass_asymmetry,
  png_compressibility, rotational_symmetry, persistence_score)
- Archive.descriptor_names round-trip
- evaluate() with non-default active_descriptors
- load_descriptor_module validation (via the CLI helper)
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import typer

from biota.cli import load_descriptor_module
from biota.search.archive import DEFAULT_DESCRIPTOR_NAMES, Archive
from biota.search.descriptors import (
    DEFAULT_DESCRIPTORS,
    REGISTRY,
    WINDOW,
    Descriptor,
    RolloutTrace,
    compute_compactness,
    compute_mass_asymmetry,
    compute_oscillation,
    compute_persistence_score,
    compute_png_compressibility,
    compute_rotational_symmetry,
    resolve_descriptors,
)
from biota.search.quality import EvaluationResult, RolloutEvaluation, evaluate

GRID = 96
STEPS = 300
TRACE_LEN = 2 * WINDOW


def _make_trace(
    *,
    com_path: np.ndarray | None = None,
    bbox_history: np.ndarray | None = None,
    final_state: np.ndarray | None = None,
) -> RolloutTrace:
    if com_path is None:
        com_path = np.full((TRACE_LEN, 2), 48.0, dtype=np.float32)
    if bbox_history is None:
        bbox_history = np.full(TRACE_LEN, 0.04, dtype=np.float32)
    if final_state is None:
        state = np.zeros((GRID, GRID), dtype=np.float32)
        state[40:50, 40:50] = 1.0
        final_state = state
    return RolloutTrace(
        com_history=com_path.astype(np.float32),
        bbox_fraction_history=bbox_history.astype(np.float32),
        gyradius_history=np.zeros(len(bbox_history), dtype=np.float32),
        final_state=final_state.astype(np.float32),
        grid_size=GRID,
        total_steps=STEPS,
    )


def _write_module(content: str) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        tmp.write(content)
        return Path(tmp.name)


# === REGISTRY ===


def test_registry_has_fifteen_built_ins() -> None:
    assert len(REGISTRY) == 15


def test_registry_contains_all_expected_names() -> None:
    expected = {
        "velocity",
        "gyradius",
        "spectral_entropy",
        "oscillation",
        "compactness",
        "mass_asymmetry",
        "png_compressibility",
        "rotational_symmetry",
        "persistence_score",
        "displacement_ratio",
        "angular_velocity",
        "growth_gradient",
        "morphological_instability",
        "activity",
        "spatial_entropy",
    }
    assert set(REGISTRY.keys()) == expected


def test_all_registry_entries_are_descriptor_instances() -> None:
    for name, d in REGISTRY.items():
        assert isinstance(d, Descriptor), f"{name} is not a Descriptor"


def test_all_registry_entries_have_callable_compute() -> None:
    for name, d in REGISTRY.items():
        assert callable(d.compute), f"{name}.compute is not callable"


def test_all_registry_entries_have_nonempty_metadata() -> None:
    for name, d in REGISTRY.items():
        assert d.name, f"{name}.name is empty"
        assert d.short_name, f"{name}.short_name is empty"
        assert d.direction_label, f"{name}.direction_label is empty"


def test_default_descriptors_are_in_registry() -> None:
    for name in DEFAULT_DESCRIPTORS:
        assert name in REGISTRY


# === resolve_descriptors ===


def test_resolve_descriptors_returns_correct_objects() -> None:
    d0, d1, d2 = resolve_descriptors(("velocity", "gyradius", "spectral_entropy"))
    assert d0 is REGISTRY["velocity"]
    assert d1 is REGISTRY["gyradius"]
    assert d2 is REGISTRY["spectral_entropy"]


def test_resolve_descriptors_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown descriptor"):
        resolve_descriptors(("velocity", "gyradius", "doesnotexist"))


def test_resolve_descriptors_error_lists_known_names() -> None:
    with pytest.raises(ValueError, match="velocity"):
        resolve_descriptors(("velocity", "gyradius", "bad"))


def test_resolve_descriptors_all_new_built_ins() -> None:
    names = ("oscillation", "compactness", "png_compressibility")
    d0, d1, d2 = resolve_descriptors(names)
    assert d0.name == "oscillation"
    assert d1.name == "compactness"
    assert d2.name == "PNG compressibility"


# === oscillation ===


def test_oscillation_zero_for_constant_bbox() -> None:
    bbox = np.full(TRACE_LEN, 0.1, dtype=np.float32)
    assert compute_oscillation(_make_trace(bbox_history=bbox)) == 0.0


def test_oscillation_higher_for_pulsing_bbox() -> None:
    bbox = np.tile([0.1, 0.5], TRACE_LEN // 2).astype(np.float32)
    val = compute_oscillation(_make_trace(bbox_history=bbox))
    assert val > 0.5


def test_oscillation_in_unit_range() -> None:
    rng = np.random.default_rng(42)
    bbox = rng.random(TRACE_LEN).astype(np.float32)
    val = compute_oscillation(_make_trace(bbox_history=bbox))
    assert 0.0 <= val <= 1.0


def test_oscillation_short_history_is_zero() -> None:
    trace = _make_trace(bbox_history=np.array([0.1], dtype=np.float32))
    assert compute_oscillation(trace) == 0.0


# === compactness ===


def test_compactness_dense_blob_is_near_one() -> None:
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[40:60, 40:60] = 1.0
    assert compute_compactness(_make_trace(final_state=state)) > 0.95


def test_compactness_scattered_mass_is_low() -> None:
    state = np.full((GRID, GRID), 0.05, dtype=np.float32)
    state[45:55, 45:55] = 1.0
    assert compute_compactness(_make_trace(final_state=state)) < 0.6


def test_compactness_zero_mass_is_zero() -> None:
    state = np.zeros((GRID, GRID), dtype=np.float32)
    assert compute_compactness(_make_trace(final_state=state)) == 0.0


def test_compactness_in_unit_range() -> None:
    rng = np.random.default_rng(7)
    state = rng.random((GRID, GRID)).astype(np.float32)
    val = compute_compactness(_make_trace(final_state=state))
    assert 0.0 <= val <= 1.0


# === mass_asymmetry ===


def test_mass_asymmetry_straight_x_mover_is_high() -> None:
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    coms[:, 1] = np.arange(TRACE_LEN) * 0.01
    val = compute_mass_asymmetry(_make_trace(com_path=coms))
    assert val == pytest.approx(1.0, abs=0.01)


def test_mass_asymmetry_diagonal_mover_is_zero() -> None:
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    coms[:, 0] = np.arange(TRACE_LEN) * 0.01
    coms[:, 1] = np.arange(TRACE_LEN) * 0.01
    val = compute_mass_asymmetry(_make_trace(com_path=coms))
    assert val == pytest.approx(0.0, abs=0.01)


def test_mass_asymmetry_stationary_is_zero() -> None:
    coms = np.full((TRACE_LEN, 2), 48.0, dtype=np.float32)
    val = compute_mass_asymmetry(_make_trace(com_path=coms))
    assert val == 0.0


def test_mass_asymmetry_in_unit_range() -> None:
    rng = np.random.default_rng(13)
    coms = rng.random((TRACE_LEN, 2)).astype(np.float32)
    val = compute_mass_asymmetry(_make_trace(com_path=coms))
    assert 0.0 <= val <= 1.0


# === png_compressibility ===


def test_png_compressibility_uniform_state_is_low() -> None:
    state = np.ones((GRID, GRID), dtype=np.float32)
    val = compute_png_compressibility(_make_trace(final_state=state))
    assert val < 0.1


def test_png_compressibility_random_state_is_high() -> None:
    rng = np.random.default_rng(99)
    state = rng.random((GRID, GRID)).astype(np.float32)
    val = compute_png_compressibility(_make_trace(final_state=state))
    assert val > 0.5


def test_png_compressibility_zero_mass_is_zero() -> None:
    state = np.zeros((GRID, GRID), dtype=np.float32)
    assert compute_png_compressibility(_make_trace(final_state=state)) == 0.0


def test_png_compressibility_in_unit_range() -> None:
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[30:50, 30:50] = 0.7
    val = compute_png_compressibility(_make_trace(final_state=state))
    assert 0.0 <= val <= 1.0


def test_png_compressibility_structured_higher_than_uniform() -> None:
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[::6, ::6] = 1.0
    val = compute_png_compressibility(_make_trace(final_state=state))
    uniform = np.ones((GRID, GRID), dtype=np.float32)
    uniform_val = compute_png_compressibility(_make_trace(final_state=uniform))
    assert val > uniform_val


# === rotational_symmetry ===


def test_rotational_symmetry_centered_disk_is_low() -> None:
    y_idx, x_idx = np.indices((GRID, GRID), dtype=np.float32)
    cy, cx = GRID / 2.0, GRID / 2.0
    radius = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    state = (radius < 15).astype(np.float32)
    val = compute_rotational_symmetry(_make_trace(final_state=state))
    assert val < 0.1


def test_rotational_symmetry_dumbbell_is_higher() -> None:
    # Two blobs far from their joint COM - mass concentrated at two opposite angles.
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[20:30, 20:30] = 1.0
    state[66:76, 66:76] = 1.0
    val = compute_rotational_symmetry(_make_trace(final_state=state))
    assert val > 0.1


def test_rotational_symmetry_zero_mass_is_zero() -> None:
    state = np.zeros((GRID, GRID), dtype=np.float32)
    assert compute_rotational_symmetry(_make_trace(final_state=state)) == 0.0


def test_rotational_symmetry_in_unit_range() -> None:
    rng = np.random.default_rng(17)
    state = rng.random((GRID, GRID)).astype(np.float32)
    val = compute_rotational_symmetry(_make_trace(final_state=state))
    assert 0.0 <= val <= 1.0


# === persistence_score ===


def test_persistence_score_stable_creature_is_low() -> None:
    coms = np.full((TRACE_LEN, 2), 48.0, dtype=np.float32)
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[40:50, 40:50] = 1.0
    val = compute_persistence_score(_make_trace(com_path=coms, final_state=state))
    assert val == pytest.approx(0.0, abs=0.01)


def test_persistence_score_drifting_creature_is_high() -> None:
    coms = np.zeros((TRACE_LEN, 2), dtype=np.float32)
    coms[WINDOW:, 0] = np.arange(WINDOW) * 2.0
    state = np.zeros((GRID, GRID), dtype=np.float32)
    state[40:60, 40:60] = 1.0
    val = compute_persistence_score(_make_trace(com_path=coms, final_state=state))
    assert val >= 1.0


def test_persistence_score_short_trace_is_zero() -> None:
    short_coms = np.zeros((WINDOW - 1, 2), dtype=np.float32)
    trace = RolloutTrace(
        com_history=short_coms,
        bbox_fraction_history=np.full(WINDOW - 1, 0.1, dtype=np.float32),
        gyradius_history=np.zeros(WINDOW - 1, dtype=np.float32),
        final_state=np.zeros((GRID, GRID), dtype=np.float32),
        grid_size=GRID,
        total_steps=STEPS,
    )
    assert compute_persistence_score(trace) == 0.0


def test_persistence_score_in_unit_range() -> None:
    rng = np.random.default_rng(23)
    coms = rng.random((TRACE_LEN, 2)).astype(np.float32) * 10
    state = rng.random((GRID, GRID)).astype(np.float32)
    val = compute_persistence_score(_make_trace(com_path=coms, final_state=state))
    assert 0.0 <= val <= 1.0


# === Archive.descriptor_names ===


def test_archive_default_descriptor_names() -> None:
    archive = Archive()
    assert archive.descriptor_names == DEFAULT_DESCRIPTOR_NAMES
    assert archive.descriptor_names == ("velocity", "gyradius", "spectral_entropy")


def test_archive_custom_descriptor_names_stored() -> None:
    names: tuple[str, str, str] = ("oscillation", "compactness", "png_compressibility")
    archive = Archive(descriptor_names=names)
    assert archive.descriptor_names == names


# === evaluate() with active_descriptors ===


def _make_eval(trace: RolloutTrace | None = None) -> EvaluationResult:
    return evaluate(
        RolloutEvaluation(
            initial_mass=100.0,
            final_mass=100.0,
            trace=trace if trace is not None else _make_trace(),
        )
    )


def test_evaluate_default_active_produces_three_descriptors() -> None:
    result = _make_eval()
    assert result.descriptors is not None
    assert len(result.descriptors) == 3


def test_evaluate_with_non_default_active_descriptors() -> None:
    active = resolve_descriptors(("oscillation", "compactness", "mass_asymmetry"))
    result = evaluate(
        RolloutEvaluation(
            initial_mass=100.0,
            final_mass=100.0,
            trace=_make_trace(),
        ),
        active_descriptors=active,
    )
    assert result.descriptors is not None
    assert len(result.descriptors) == 3
    for d in result.descriptors:
        assert 0.0 <= d <= 1.0


def test_evaluate_active_descriptors_differ_from_default() -> None:
    active = resolve_descriptors(("oscillation", "compactness", "mass_asymmetry"))
    default_result = _make_eval()
    active_result = evaluate(
        RolloutEvaluation(
            initial_mass=100.0,
            final_mass=100.0,
            trace=_make_trace(),
        ),
        active_descriptors=active,
    )
    assert default_result.descriptors is not None
    assert active_result.descriptors is not None
    assert default_result.descriptors != active_result.descriptors


# === _load_descriptor_module ===


def test_load_descriptor_module_valid() -> None:
    path = _write_module(
        "from biota.search.descriptors import Descriptor, RolloutTrace\n"
        "def _c(trace: RolloutTrace) -> float: return 0.5\n"
        "DESCRIPTORS = [Descriptor(name='test_d', short_name='t', direction_label='more', compute=_c)]\n"
    )
    load_descriptor_module(path)
    assert "test_d" in REGISTRY
    del REGISTRY["test_d"]


def test_load_descriptor_module_missing_descriptors_list() -> None:
    path = _write_module("x = 1\n")
    with pytest.raises(typer.BadParameter, match="DESCRIPTORS"):
        load_descriptor_module(path)


def test_load_descriptor_module_not_a_list() -> None:
    path = _write_module("DESCRIPTORS = 'not a list'\n")
    with pytest.raises(typer.BadParameter, match="list"):
        load_descriptor_module(path)


def test_load_descriptor_module_wrong_element_type() -> None:
    path = _write_module("DESCRIPTORS = ['not a descriptor']\n")
    with pytest.raises(typer.BadParameter, match="Descriptor"):
        load_descriptor_module(path)


def test_load_descriptor_module_bad_file_raises() -> None:
    with pytest.raises(typer.BadParameter):
        load_descriptor_module(Path("/nonexistent/path/to/module.py"))

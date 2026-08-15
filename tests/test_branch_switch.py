import numpy as np

from threebody_atlas.branch_switch import branch_amplitude, scaled_branch_direction


def test_scaled_branch_amplitude_matches_seed_by_construction() -> None:
    reference = np.asarray([0.1, -0.2, 1.0, 0.0, 0.4, 1.2, -0.7, 2.0])
    direction = np.asarray([1.0, 2.0, -0.5, 0.3, -1.0, 0.4, 0.8, -0.2])
    scale, unit = scaled_branch_direction(reference, direction)
    target = -1.7e-3
    seeded = reference + target * scale * unit
    assert abs(branch_amplitude(seeded, reference, direction) - target) < 1e-15
    assert abs(np.linalg.norm(unit) - 1.0) < 1e-15


def test_branch_amplitude_changes_sign_with_direction() -> None:
    reference = np.zeros(8)
    direction = np.arange(1.0, 9.0)
    displaced = reference + 1e-3 * direction
    assert branch_amplitude(displaced, reference, direction) > 0.0
    assert branch_amplitude(displaced, reference, -direction) < 0.0

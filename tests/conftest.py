"""Shared fixtures for all tests."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _reset_cli_globals():
    """Reset __main__ module globals after every test.

    main() mutates _use_color, _show_dispatch, _show_mode_sensitivity.
    Without reset, tests that set these to True leak into subsequent tests
    that call format_tree() or format_leaves() directly.
    """
    import decomp_magician.__main__ as m

    old_color = m._use_color
    old_dispatch = m._show_dispatch
    old_mode = m._show_mode_sensitivity
    yield
    m._use_color = old_color
    m._show_dispatch = old_dispatch
    m._show_mode_sensitivity = old_mode


# ---------------------------------------------------------------------------
# Fixtures that discover ops by property at runtime, so tests don't hardcode
# assumptions about which ops have which classification in a given PyTorch
# version.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def inductor_kept_op():
    """An op that is in the decomposition table but kept by Inductor."""
    from torch._decomp import decomposition_table
    from decomp_magician.classify import _build_inductor_kept

    kept_names = _build_inductor_kept()
    for op in decomposition_table:
        if op.name() in kept_names:
            return op
    pytest.skip("No inductor-kept ops in this PyTorch build")


@pytest.fixture(scope="session")
def non_inductor_kept_decomposable_op():
    """An op that has a decomposition and is NOT inductor-kept."""
    from torch._decomp import decomposition_table
    from decomp_magician.classify import _build_inductor_kept

    kept_names = _build_inductor_kept()
    for op in decomposition_table:
        if op.name() not in kept_names:
            return op
    pytest.skip("All decomposable ops are inductor-kept in this PyTorch build")


@pytest.fixture(scope="session")
def dotted_overload_op():
    """An aten op with a non-default overload (e.g. 'add.Tensor')."""
    from decomp_magician.resolve import resolve_op

    # These are stable across PyTorch versions
    candidates = ["add.Tensor", "softmax.int", "sum.dim_IntList"]
    for name in candidates:
        result = resolve_op(name)
        if isinstance(result, torch._ops.OpOverload):
            return name, result
    pytest.skip("No dotted overload ops found")

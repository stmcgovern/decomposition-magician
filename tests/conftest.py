"""Shared fixtures for all tests."""

import pytest


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

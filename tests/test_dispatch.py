"""Tests for dispatch table introspection."""

import torch

from decomp_magician.dispatch import (
    DispatchEntry,
    get_dispatch_info,
    get_dispatch_info_cached,
    format_dispatch_short,
    format_dispatch_detail,
)


class TestDispatchEntry:
    def test_fallthrough(self):
        e = DispatchEntry(key="AutogradCPU", tag="", is_fallthrough=True)
        assert e.is_fallthrough
        assert not e.is_redispatch

    def test_autograd_kernel(self):
        e = DispatchEntry(key="AutogradCPU", tag="autograd kernel", is_fallthrough=False)
        assert e.is_redispatch
        assert not e.is_terminal

    def test_math_kernel(self):
        e = DispatchEntry(key="AutogradCPU", tag="math kernel", is_fallthrough=False)
        assert e.is_terminal
        assert not e.is_redispatch

    def test_redispatch_and_terminal_mutually_exclusive(self):
        """For any tag, exactly one of is_redispatch/is_terminal can be true."""
        for tag in ("autograd kernel", "math kernel", "", "other kernel"):
            for ft in (True, False):
                e = DispatchEntry(key="X", tag=tag, is_fallthrough=ft)
                assert not (e.is_redispatch and e.is_terminal), f"both true for tag={tag!r}, ft={ft}"


class TestGetDispatchInfo:
    def test_mm_autograd_type_is_autograd_kernel(self):
        """mm has [autograd kernel] at AutogradCPU (verified from raw table)."""
        op = torch.ops.aten.mm.default
        info = get_dispatch_info(op)
        assert info.autograd_type == "autograd_kernel"

    def test_mm_adiov_is_fallthrough(self):
        """mm has ADIOV=fallthrough (verified from raw table)."""
        op = torch.ops.aten.mm.default
        info = get_dispatch_info(op)
        assert info.has_adiov is False

    def test_add_inplace_has_adiov(self):
        """add_ has a non-fallthrough ADIOV [kernel] (verified from raw table)."""
        op = torch.ops.aten.add_.Tensor
        info = get_dispatch_info(op)
        assert info.has_adiov is True

    def test_add_inplace_mode_sensitive(self):
        """add_ has [autograd kernel] at AutogradCPU → mode_sensitive."""
        op = torch.ops.aten.add_.Tensor
        info = get_dispatch_info(op)
        assert info.mode_sensitive is True

    def test_cross_verify_against_raw_table(self):
        """Cross-verify get_dispatch_info against raw _dispatch_dump_table.

        This eliminates circularity: we parse the raw table string independently
        and check that our structured analysis agrees.
        """
        for op_name, op in [
            ("aten::mm", torch.ops.aten.mm.default),
            ("aten::add.Tensor", torch.ops.aten.add.Tensor),
            ("aten::add_.Tensor", torch.ops.aten.add_.Tensor),
        ]:
            raw = torch._C._dispatch_dump_table(op_name)
            info = get_dispatch_info(op)

            # Check ADIOV: find the ADInplaceOrView line
            adiov_line = ""
            for line in raw.strip().split("\n"):
                if "ADInplaceOrView:" in line:
                    adiov_line = line.strip()
                    break

            if adiov_line:
                raw_is_fallthrough = "fallthrough" in adiov_line.lower()
                assert info.has_adiov == (not raw_is_fallthrough), \
                    f"{op_name}: has_adiov={info.has_adiov} but raw fallthrough={raw_is_fallthrough}"

    def test_cached_returns_same(self):
        op = torch.ops.aten.mm.default
        info1 = get_dispatch_info_cached(op)
        info2 = get_dispatch_info_cached(op)
        assert info1 is info2


class TestFormatDispatch:
    def test_short_format_mm(self):
        info = get_dispatch_info(torch.ops.aten.mm.default)
        short = format_dispatch_short(info)
        assert "AG:redispatch" in short  # mm has autograd kernel → "redispatch"
        assert "ADIOV" not in short  # mm has no ADIOV

    def test_short_format_add_inplace(self):
        info = get_dispatch_info(torch.ops.aten.add_.Tensor)
        short = format_dispatch_short(info)
        assert "ADIOV:yes" in short

    def test_detail_format(self):
        info = get_dispatch_info(torch.ops.aten.mm.default)
        detail = format_dispatch_detail(info)
        assert "dispatch:" in detail
        assert "AutogradCPU:" in detail

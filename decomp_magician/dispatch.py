"""Dispatch table introspection: per-op kernel entries and mode sensitivity.

Cross-references PyTorch's internal dispatch tables to classify ops by their
autograd-level behavior and ADInplaceOrView kernel presence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch._ops import OpOverload


@dataclass(frozen=True)
class DispatchEntry:
    """A single dispatch table entry for one key."""
    key: str
    tag: str  # e.g. "math kernel", "autograd kernel", ""
    is_fallthrough: bool

    @property
    def is_redispatch(self) -> bool:
        return not self.is_fallthrough and "kernel" in self.tag and self.tag != "math kernel"

    @property
    def is_terminal(self) -> bool:
        return not self.is_fallthrough and "math kernel" in self.tag


@dataclass(frozen=True)
class DispatchInfo:
    """Dispatch table analysis for a single op."""
    op_name: str
    autograd_entry: DispatchEntry | None  # AutogradCPU entry
    adiov_entry: DispatchEntry | None  # ADInplaceOrView entry
    dense_entry: DispatchEntry | None  # Dense/CPU entry
    autograd_type: str  # "autograd_kernel", "math_kernel", "fallthrough", "other", "none"
    has_adiov: bool  # non-fallthrough ADIOV kernel
    mode_sensitive: bool  # has non-fallthrough autograd entry (any type)

    @property
    def differs_under_inference_mode(self) -> bool:
        """Op behaves differently under inference_mode vs no_grad (has ADIOV kernel)."""
        return self.has_adiov


_AUTOGRAD_KEY = "AutogradCPU"
_ADIOV_KEY = "ADInplaceOrView"
_DENSE_KEY = "CPU"

# Keys to extract from dispatch tables
_KEYS_OF_INTEREST = [_AUTOGRAD_KEY, _ADIOV_KEY, _DENSE_KEY]


def _parse_dispatch_table(op_name: str) -> dict[str, DispatchEntry]:
    """Parse dispatch table into dict of key -> DispatchEntry."""
    try:
        raw = torch._C._dispatch_dump_table(op_name)
    except Exception:
        return {}

    entries = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("Registered") or line.startswith("catchall"):
            continue
        match = re.match(r"^(\S+?):\s*(.*)", line)
        if match:
            key_name = match.group(1)
            rest = match.group(2)
            tag_match = re.search(r"\[([^\]]+)\]\s*$", rest)
            tag = tag_match.group(1) if tag_match else ""
            entries[key_name] = DispatchEntry(
                key=key_name,
                tag=tag,
                is_fallthrough="fallthrough" in rest.lower(),
            )
    return entries


def _classify_autograd_type(entry: DispatchEntry | None) -> str:
    """Classify the autograd behavior from an AutogradCPU entry."""
    if entry is None:
        return "none"
    if entry.is_fallthrough:
        return "fallthrough"
    if "math kernel" in entry.tag:
        return "math_kernel"
    if "autograd kernel" in entry.tag:
        return "autograd_kernel"
    return "other"


def _build_dispatch_info(op_name: str) -> DispatchInfo:
    """Build dispatch info from an op name string."""
    entries = _parse_dispatch_table(op_name)

    ag = entries.get(_AUTOGRAD_KEY)
    adiov = entries.get(_ADIOV_KEY)
    dense = entries.get(_DENSE_KEY)

    ag_type = _classify_autograd_type(ag)
    has_adiov = adiov is not None and not adiov.is_fallthrough
    mode_sensitive = ag_type not in ("none", "fallthrough")

    return DispatchInfo(
        op_name=op_name,
        autograd_entry=ag,
        adiov_entry=adiov,
        dense_entry=dense,
        autograd_type=ag_type,
        has_adiov=has_adiov,
        mode_sensitive=mode_sensitive,
    )


def get_dispatch_info(op: OpOverload) -> DispatchInfo:
    """Get dispatch table analysis for a single op."""
    return _build_dispatch_info(op.name())


def get_dispatch_info_by_name(op_name: str) -> DispatchInfo:
    """Get dispatch table analysis from an aten:: op name string."""
    return _build_dispatch_info(op_name)


# Cache for bulk lookups (e.g. tree walks)
_dispatch_cache: dict[str, DispatchInfo] = {}


def get_dispatch_info_cached(op: OpOverload) -> DispatchInfo:
    """Cached version of get_dispatch_info for tree walks."""
    name = op.name()
    if name not in _dispatch_cache:
        _dispatch_cache[name] = get_dispatch_info(op)
    return _dispatch_cache[name]


def format_dispatch_short(info: DispatchInfo) -> str:
    """Short annotation string for dispatch info: AG:type, ADIOV:yes/no."""
    parts = []
    ag_labels = {
        "autograd_kernel": "AG:redispatch",
        "math_kernel": "AG:terminal",
        "fallthrough": "AG:fallthrough",
        "other": "AG:other",
        "none": "AG:none",
    }
    parts.append(ag_labels.get(info.autograd_type, f"AG:{info.autograd_type}"))
    if info.has_adiov:
        parts.append("ADIOV:yes")
    return ", ".join(parts)


def format_dispatch_detail(info: DispatchInfo) -> str:
    """Multi-line dispatch info for verbose output."""
    lines = [f"  dispatch: {info.autograd_type}"]
    if info.autograd_entry:
        lines.append(f"    AutogradCPU: [{info.autograd_entry.tag}]"
                      f"{' (fallthrough)' if info.autograd_entry.is_fallthrough else ''}")
    if info.adiov_entry:
        lines.append(f"    ADInplaceOrView: [{info.adiov_entry.tag}]"
                      f"{' (fallthrough)' if info.adiov_entry.is_fallthrough else ''}")
    if info.dense_entry:
        lines.append(f"    CPU: [{info.dense_entry.tag}]"
                      f"{' (fallthrough)' if info.dense_entry.is_fallthrough else ''}")
    return "\n".join(lines)

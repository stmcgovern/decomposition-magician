"""Op name resolution: user-friendly name → OpOverload."""

from __future__ import annotations

import torch
from torch._ops import OpOverload, OpOverloadPacket


def resolve_op(name: str) -> OpOverload | list[str]:
    """
    Resolve a user-provided name to an OpOverload.

    Resolution order:
      1. Exact: "aten.addcmul.default" → direct lookup
      2. Default overload: "aten.addcmul" → try .default, then first overload
      3. Namespace prefix: "addcmul" → try aten.addcmul
      4. Substring search: "batch_norm" → list matching ops

    Returns the resolved OpOverload, or a list of candidate name strings
    if the input is ambiguous or not found.
    """
    # Normalize C++ format: "aten::addcmul.default" → "aten.addcmul.default"
    name = name.replace("::", ".")

    # 1. Try exact match: "aten.addcmul.default"
    result = _try_exact(name)
    if result is not None:
        return result

    # 2. Try as namespace.opname (no overload): "aten.addcmul"
    if "." in name:
        parts = name.split(".", 1)
        result = _try_packet(parts[0], parts[1])
        if isinstance(result, OpOverload):
            return result
        if isinstance(result, list):
            return result

    # 3. Try with aten prefix: "addcmul" → "aten.addcmul"
    if "." not in name:
        result = _try_packet("aten", name)
        if isinstance(result, OpOverload):
            return result
        if isinstance(result, list):
            return result

    # 4. Substring search across aten namespace
    return _substring_search(name)


def _try_exact(name: str) -> OpOverload | None:
    """Try to resolve a fully qualified name like 'aten.addcmul.default'."""
    parts = name.split(".")
    if len(parts) != 3:
        return None
    ns, opname, overload = parts
    try:
        packet = getattr(getattr(torch.ops, ns), opname)
        if isinstance(packet, OpOverloadPacket):
            return getattr(packet, overload)
    except AttributeError:
        pass
    return None


def _try_packet(ns: str, opname: str) -> OpOverload | list[str] | None:
    """Try to resolve namespace.opname, picking the best overload."""
    try:
        packet = getattr(getattr(torch.ops, ns), opname)
    except AttributeError:
        return None
    if not isinstance(packet, OpOverloadPacket):
        return None

    overloads = packet.overloads()
    if not overloads:
        return None

    # Prefer .default if it exists
    if "default" in overloads:
        return getattr(packet, "default")

    # Single overload — use it
    if len(overloads) == 1:
        return getattr(packet, overloads[0])

    # Multiple non-default overloads — ambiguous
    return [f"{ns}.{opname}.{ol}" for ol in overloads]


def _substring_search(query: str) -> list[str]:
    """Search aten namespace for ops whose name contains the query."""
    query_lower = query.lower()
    matches = []
    for attr_name in dir(torch.ops.aten):
        if query_lower in attr_name.lower():
            try:
                packet = getattr(torch.ops.aten, attr_name)
                if isinstance(packet, OpOverloadPacket):
                    for ol in packet.overloads():
                        matches.append(f"aten.{attr_name}.{ol}")
            except (AttributeError, RuntimeError):
                pass
    return matches

"""Op name resolution: user-friendly name → OpOverload."""

from __future__ import annotations

import torch
from torch._ops import OpOverload, OpOverloadPacket

# Namespaces to search when resolving bare names
_NAMESPACES = ("aten", "prims")


def resolve_op(name: str) -> OpOverload | list[str]:
    """
    Resolve a user-provided name to an OpOverload.

    Resolution order:
      1. Exact: "aten.addcmul.default" → direct lookup
      2. Default overload: "aten.addcmul" → try .default, then first overload
      3. Namespace prefix: "addcmul" → try aten.addcmul, prims.addcmul
         Also: "logsumexp.dim_IntList" → try aten.logsumexp.dim_IntList
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

    # 3. Try with namespace prefix: "addcmul" → "aten.addcmul", "prims.addcmul"
    #    Also handles "opname.overload" → "aten.opname.overload"
    if "." not in name:
        for ns in _NAMESPACES:
            result = _try_packet(ns, name)
            if isinstance(result, OpOverload):
                return result
            if isinstance(result, list):
                return result
    elif name.count(".") == 1:
        # Could be "opname.overload" (e.g. "logsumexp.dim_IntList")
        opname, overload = name.split(".")
        for ns in _NAMESPACES:
            result = _try_exact(f"{ns}.{opname}.{overload}")
            if result is not None:
                return result

    # 4. Substring search across all namespaces
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

    # Prefer .default if it exists and is a real operator
    if "default" in overloads:
        op = getattr(packet, "default")
        if _is_real_op(op):
            return op

    # Filter out _out variants and broken ops
    primary = [ol for ol in overloads
               if not ol.endswith("_out") and ol != "out"
               and _is_real_op(getattr(packet, ol))]

    if len(primary) == 1:
        return getattr(packet, primary[0])

    # Multiple valid overloads — pick first, caller prints a note
    if primary:
        return getattr(packet, primary[0])

    # Try _out variants as last resort
    out_overloads = [ol for ol in overloads if _is_real_op(getattr(packet, ol))]
    if out_overloads:
        return getattr(packet, out_overloads[0])

    return None


def _is_real_op(op: OpOverload) -> bool:
    """Check if an OpOverload is backed by a real C++ operator.

    Some overloads (e.g. aten.add.default, aten.add.int) exist as Python
    stubs but have no C++ dispatch entry, causing RuntimeError on use.
    """
    try:
        torch._C._dispatch_has_kernel_for_dispatch_key(
            op.name(), torch._C.DispatchKey.CompositeImplicitAutograd,
        )
        return True
    except RuntimeError:
        return False


def _substring_search(query: str) -> list[str]:
    """Search all namespaces for ops whose qualified name contains the query."""
    query_lower = query.lower()
    matches = []
    for ns_name in _NAMESPACES:
        try:
            ns = getattr(torch.ops, ns_name)
        except AttributeError:
            continue
        for attr_name in dir(ns):
            try:
                packet = getattr(ns, attr_name)
                if not isinstance(packet, OpOverloadPacket):
                    continue
                for ol in packet.overloads():
                    full_name = f"{ns_name}.{attr_name}.{ol}"
                    if query_lower in full_name.lower():
                        matches.append(full_name)
            except (AttributeError, RuntimeError):
                pass
    return matches

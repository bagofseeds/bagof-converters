"""
Lazy registry keys and lazy class attributes, for optional libraries.

A registry (a dict mapping hints to values, filled with
`registry[hint] = value`) can be keyed by a [`ForwardRef`][typing.ForwardRef]
instead of a real object. The key is kept pending and moved into the
registry -- keyed by the object it names -- once that object has been
imported by someone else. Nothing here ever imports a module or evaluates
a string to resolve a key.

!!! note
    This module only depends on the standard library and
    `typing_extensions`, so that it can move to `bagof.core.magic` and be
    shared by every registry-based bag.
"""

__all__ = [
    "is_forward_ref",
    "defer",
    "resolve_pending",
    "pending",
    "find_name",
    "has_module",
    "import_name",
    "lazy_import",
]

# stdlib
import importlib
import importlib.util
import sys
import types
import typing

# dependencies
import typing_extensions as tx

_FORWARD_REFS = tuple({typing.ForwardRef, tx.ForwardRef})
"""The forward-reference types that make a lazy registry key."""

_MISSING = object()
"""Marker for a name that cannot be resolved (yet)."""

_PENDING: tx.List[tx.Tuple[dict, str, str, tx.Any]] = []
"""
Registrations keyed by a forward reference whose target is not imported
yet, as `(registry, context_module, "dotted.name", value)`. A hint or value
of a type from a module that was never imported cannot reach a registry,
so nothing is lost by waiting.
"""


def is_forward_ref(obj: tx.Any) -> bool:
    """Whether `obj` is a [`ForwardRef`][typing.ForwardRef]."""
    return isinstance(obj, _FORWARD_REFS)


def defer(registry: dict, ref: tx.Any, value: tx.Any, context: str) -> None:
    """
    Register `value` in `registry` under the object `ref` names, once that
    object can be found -- see [`find_name`][].

    Parameters
    ----------
    registry : dict
        The registry to fill.
    ref : ForwardRef
        The forward reference. Its `module=` (if any) takes precedence
        over `context`.
    value : Any
        The value to register.
    context : str
        The module the reference is relative to, usually the module of
        the registering class.
    """
    context = getattr(ref, "__forward_module__", None) or context
    name = ref.__forward_arg__
    # Compare names, not refs: `ForwardRef` equality also involves fields
    # that vary across Python versions. A newer entry replaces an older one.
    _PENDING[:] = [
        entry for entry in _PENDING
        if entry[0] is not registry or entry[1:3] != (context, name)
    ]
    _PENDING.append((registry, context, name, value))


def resolve_pending() -> None:
    """Move the pending entries that can now be found into their registry."""
    if not _PENDING:
        return
    for entry in list(_PENDING):
        registry, context, name, value = entry
        obj = find_name(name, context)
        if obj is _MISSING:
            continue
        _PENDING.remove(entry)
        registry[obj] = value


def pending(
    registry: tx.Optional[dict] = None,
) -> tx.List[tx.Tuple[str, str, tx.Any]]:
    """
    The pending entries (of `registry`, or of all registries), as
    `(context_module, "dotted.name", value)`. Meant for debugging.
    """
    return [
        entry[1:] for entry in _PENDING
        if registry is None or entry[0] is registry
    ]


def find_name(name: str, context: tx.Optional[str] = None) -> tx.Any:
    """
    The object a dotted name refers to, if it is already imported. Never
    imports anything.

    The name is first looked up relative to the `context` module (its
    first part being a global of that module), then as an absolute dotted
    path (the longest prefix found in [`sys.modules`][], followed by
    attributes).

    Returns
    -------
    Any
        The object, or a private marker if it cannot be found (yet).
    """
    parts = name.split(".")
    # 1. relative to the context module
    module = sys.modules.get(context) if context else None
    if module is not None:
        obj = _walk(module, parts)
        if obj is not _MISSING:
            return obj
    # 2. absolute
    for i in range(len(parts), 0, -1):
        module = sys.modules.get(".".join(parts[:i]))
        if module is not None:
            return _walk(module, parts[i:])
    return _MISSING


def _walk(obj: tx.Any, attrs: tx.Iterable[str]) -> tx.Any:
    """Walk attributes without triggering a module-level `__getattr__`."""
    for attr in attrs:
        if isinstance(obj, types.ModuleType):
            # Read the module dict directly: a module-level `__getattr__`
            # (PEP 562) may import on access.
            obj = vars(obj).get(attr, _MISSING)
        else:
            obj = getattr(obj, attr, _MISSING)
        if obj is _MISSING:
            # Not imported yet, or still initialising.
            break
    return obj


def has_module(name: str) -> bool:
    """Whether a module can be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover
        return False


def import_name(name: str) -> tx.Any:
    """Import the object named by a fully-qualified dotted name."""
    first, *rest = name.split(".")
    obj = importlib.import_module(first)
    for attr in rest:
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            # A submodule that has not been imported yet.
            obj = importlib.import_module(f"{obj.__name__}.{attr}")
    return obj


class lazy_import:
    """
    Class attribute holding an object that is only imported on first access,
    so that defining a class for an optional library does not import it.

    !!! example
        ```python
        class ToDaskArray(ArrayConverter):
            DEFAULT = lazy_import("dask.array.Array")
        ```
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, obj: tx.Any, owner: tx.Any = None) -> tx.Any:
        if "value" not in self.__dict__:
            self.value = import_name(self.name)
        return self.value

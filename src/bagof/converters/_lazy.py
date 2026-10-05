"""
Lazy registry keys and lazy class attributes, for optional libraries.

A registry (a dict mapping hints to values, filled with
`registry[hint] = value`) can be keyed by a [`ForwardRef`][typing.ForwardRef]
instead of a real object. The key is kept pending and moved into the
registry -- keyed by the object it names -- once that object has been
imported by someone else. Nothing here ever imports a module or evaluates
a string to resolve a key.

Pending keys are only retried when they may have become resolvable: when
[`sys.modules`][] has changed size since the last attempt, when a key was
added, or when an attempt stopped inside a module that was still
initialising (its dict may grow without `sys.modules` changing). Otherwise
-- the steady state -- a lookup costs a length check and never rescans.
An object bound later by other means (e.g. monkeypatching a module that
had finished importing) is picked up by `resolve_pending(force=True)`.

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


class _Entry:
    """A pending registration."""

    __slots__ = ("registry", "context", "name", "value")

    def __init__(
        self,
        registry: dict,
        context: str,
        name: str,
        value: tx.Any,
    ) -> None:
        self.registry = registry
        self.context = context
        self.name = name
        self.value = value


_PENDING: tx.List[_Entry] = []
"""
All pending registrations, in registration order. A hint or value of a
type from a module that was never imported cannot reach a registry, so
nothing is lost by waiting.
"""

_SEEN = -1
"""The size of [`sys.modules`][] at the last full attempt."""

_DIRTY = False
"""Whether the pending entries must be retried regardless of `_SEEN`."""


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
    global _DIRTY
    context = getattr(ref, "__forward_module__", None) or context
    name = ref.__forward_arg__
    # Compare names, not refs: `ForwardRef` equality also involves fields
    # that vary across Python versions. A newer entry replaces an older one.
    for entry in list(_PENDING):
        if (
            entry.registry is registry
            and entry.context == context
            and entry.name == name
        ):
            _PENDING.remove(entry)
    _PENDING.append(_Entry(registry, context, name, value))
    _DIRTY = True


def resolve_pending(force: bool = False) -> None:
    """
    Move the pending entries that can now be found into their registry,
    in registration order (so the last registration still wins).

    Does nothing unless something may have changed since the last attempt
    (see the module docstring), or `force` is true.
    """
    global _SEEN, _DIRTY
    if not _PENDING:
        return
    if not (force or _DIRTY or len(sys.modules) != _SEEN):
        return
    _SEEN, _DIRTY = len(sys.modules), False
    for entry in list(_PENDING):
        obj, settled = _find(entry.name, entry.context)
        if obj is _MISSING:
            _DIRTY = _DIRTY or not settled
            continue
        try:
            _PENDING.remove(entry)
        except ValueError:  # pragma: no cover - another thread got it
            continue
        entry.registry[obj] = entry.value


def pending(
    registry: tx.Optional[dict] = None,
) -> tx.List[tx.Tuple[str, str, tx.Any]]:
    """
    The pending entries (of `registry`, or of all registries), as
    `(context_module, "dotted.name", value)`. Meant for debugging.
    """
    return [
        (entry.context, entry.name, entry.value) for entry in _PENDING
        if registry is None or entry.registry is registry
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
    return _find(name, context)[0]


def _find(name: str, context: tx.Optional[str]) -> tx.Tuple[tx.Any, bool]:
    """
    [`find_name`][], and whether a miss is settled: `False` if a lookup
    stopped inside a module that may still grow without
    [`sys.modules`][] changing (initialising, or built by hand).
    """
    parts = name.split(".")
    settled = True
    # 1. relative to the context module
    module = sys.modules.get(context) if context else None
    if module is not None:
        obj, settled = _walk(module, parts)
        if obj is not _MISSING:
            return obj, True
    # 2. absolute
    for i in range(len(parts), 0, -1):
        module = sys.modules.get(".".join(parts[:i]))
        if module is not None:
            obj, ok = _walk(module, parts[i:])
            return obj, settled and ok
    return _MISSING, settled


def _walk(obj: tx.Any, attrs: tx.Iterable[str]) -> tx.Tuple[tx.Any, bool]:
    """
    Walk attributes without triggering a module-level `__getattr__`.
    Also returns whether a miss is settled (see [`_find`][]).
    """
    for attr in attrs:
        if isinstance(obj, types.ModuleType):
            # Read the module dict directly: a module-level `__getattr__`
            # (PEP 562) may import on access.
            found = vars(obj).get(attr, _MISSING)
            if found is _MISSING:
                return _MISSING, not _initialising(obj)
        else:
            found = getattr(obj, attr, _MISSING)
            if found is _MISSING:
                return _MISSING, True
        obj = found
    return obj, True


def _initialising(module: types.ModuleType) -> bool:
    """Whether a module may still grow: still executing, or hand-made."""
    # importlib flags a module's spec while executing it, and reads the
    # flag the same way (`importlib._bootstrap._handle_fromlist`).
    spec = getattr(module, "__spec__", None)
    return spec is None or bool(getattr(spec, "_initializing", False))


def has_module(name: str) -> bool:
    """Whether a module can be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover
        return False


def _resolve_name(name: str) -> tx.Any:
    """
    A copy of [`pkgutil.resolve_name`][] (validation left out): the
    original is 3.9+, and importing `pkgutil` (plus compiling its pattern)
    costs about 0.5 ms on first use.
    """
    modname, colon, qualname = name.partition(":")
    parts = qualname.split(".") if qualname else []
    if not colon:
        # no colon: find the package boundary by trying to import
        parts = name.split(".")
        modname = parts.pop(0)
        importlib.import_module(modname)
        while parts:
            try:
                importlib.import_module(f"{modname}.{parts[0]}")
            except ImportError:
                break
            modname += "." + parts.pop(0)
    obj = importlib.import_module(modname)
    for part in parts:
        obj = getattr(obj, part)
    return obj


def import_name(name: str) -> tx.Any:
    """
    Import an object by name, with [`pkgutil.resolve_name`][] semantics.

    * `"pkg.mod:obj.attr"` imports `pkg.mod` and walks `obj.attr`;
    * `"pkg.mod"` returns the module `pkg.mod` itself;
    * `"pkg:mod"` returns the attribute `mod` of `pkg`, importing the
      submodule `pkg.mod` if `pkg` does not bind it (unlike
      `resolve_name`, which raises);
    * a dotted name without a colon imports the longest importable
      prefix and walks the rest.
    """
    try:
        return _resolve_name(name)
    except AttributeError:
        modname, colon, qualname = name.partition(":")
        if not colon:
            raise
    # an attribute of a module may be one of its unimported submodules
    obj = importlib.import_module(modname)
    for attr in qualname.split("."):
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            if not hasattr(obj, "__path__"):
                raise  # not a package: there is no submodule to import
            obj = importlib.import_module(f"{obj.__name__}.{attr}")
    return obj


class lazy_import:
    """
    Class attribute holding an object that is only imported on first access,
    so that defining a class for an optional library does not import it.

    On first access, the descriptor replaces itself with the imported value
    on the class(es) it was defined in, so later reads are plain class
    attributes. A value that is itself a descriptor (e.g. a function, which
    would become a method) is not put on the class; the descriptor keeps
    returning it instead.

    !!! example
        ```python
        class ToDaskArray(ArrayConverter):
            DEFAULT = lazy_import("dask.array:Array")
        ```
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.owners: tx.List[tx.Tuple[type, str]] = []

    def __set_name__(self, owner: type, name: str) -> None:
        # Called once per name: `A = B = lazy_import(...)` binds two.
        self.owners.append((owner, name))

    def __get__(self, obj: tx.Any, owner: tx.Any = None) -> tx.Any:
        try:
            value = self.value
        except AttributeError:
            value = self.value = import_name(self.name)
        if not hasattr(type(value), "__get__"):
            # Set on the defining class, not on `owner` (which may be a
            # subclass), and only where this descriptor is still in place.
            # Idempotent, so concurrent first accesses are harmless.
            for cls, attr in self.owners:
                if vars(cls).get(attr) is self:
                    setattr(cls, attr, value)
        return value

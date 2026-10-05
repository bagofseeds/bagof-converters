"""Tests for the converter-agnostic lazy machinery (#55)."""

# stdlib
import importlib.machinery
import sys
import threading
import types

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.converters import _lazy
from bagof.converters._lazy import import_name, lazy_import


@pytest.fixture(autouse=True)
def _restore_pending() -> tx.Iterator[None]:
    """Drop the pending entries a test leaves behind."""
    pending = list(_lazy._PENDING)
    yield
    _lazy._PENDING[:] = pending
    _lazy._DIRTY = True


@pytest.fixture
def pkg(tmp_path: tx.Any, monkeypatch: tx.Any) -> str:
    """An on-disk package `_bagof_pkg` with an unimported submodule."""
    root = tmp_path / "_bagof_pkg"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "sub.py").write_text(
        "class Outer:\n    class Inner:\n        pass\nX = 1\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("_bagof_pkg", "_bagof_pkg.sub"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return "_bagof_pkg"


def _module(name: str, monkeypatch: tx.Any, spec: bool = True) -> tx.Any:
    """A module in `sys.modules`, finished initialising if `spec`."""
    module = types.ModuleType(name)
    if spec:
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    monkeypatch.setitem(sys.modules, name, module)
    return module


# --- import_name ------------------------------------------------------


def test_import_name_colon_walks_the_object(pkg: str) -> None:
    sub = import_name(f"{pkg}.sub")
    assert import_name(f"{pkg}.sub:Outer.Inner") is sub.Outer.Inner
    assert import_name(f"{pkg}.sub:X") == 1


def test_import_name_dotted_module_is_the_leaf(pkg: str) -> None:
    sub = import_name(f"{pkg}.sub")
    assert isinstance(sub, types.ModuleType)
    assert sub.__name__ == f"{pkg}.sub"


def test_import_name_colon_submodule_is_imported(pkg: str) -> None:
    """`"pkg:mod"` imports the submodule when `pkg` does not bind it."""
    assert f"{pkg}.sub" not in sys.modules
    sub = import_name(f"{pkg}:sub")
    assert sub is sys.modules[f"{pkg}.sub"]
    assert import_name(f"{pkg}:sub.X") == 1


def test_import_name_dotted_fallback_walks_attributes(pkg: str) -> None:
    assert import_name(f"{pkg}.sub.Outer.Inner").__name__ == "Inner"


def test_import_name_missing_attribute_raises(pkg: str) -> None:
    with pytest.raises(AttributeError):
        import_name(f"{pkg}.sub:Nope")
    with pytest.raises(ImportError):
        import_name(f"{pkg}:nope")


@pytest.mark.parametrize(
    "spec",
    ["sub", "sub:X", "sub:Outer.Inner", "sub.Outer.Inner", "sub.X", "", ":"],
)
def test_resolve_name_copy_matches_pkgutil(pkg: str, spec: str) -> None:
    """Our copy agrees with `pkgutil.resolve_name`."""
    if sys.version_info < (3, 9):
        pytest.skip("pkgutil.resolve_name is 3.9+")
    import pkgutil

    name = f"{pkg}.{spec}" if spec and spec != ":" else pkg + spec
    assert _lazy._resolve_name(name) is pkgutil.resolve_name(name)


# --- lazy_import ------------------------------------------------------


def test_lazy_import_replaces_itself() -> None:
    class A:
        X = lazy_import("json:dumps.__name__")
        MOD = lazy_import("json")

    assert isinstance(vars(A)["MOD"], lazy_import)
    import json

    assert A.MOD is json
    assert vars(A)["MOD"] is json  # a plain class attribute now
    assert A().X == "dumps"  # instance access
    assert vars(A)["X"] == "dumps"


def test_lazy_import_subclass_access_first_sets_the_base() -> None:
    import json

    class A:
        MOD = lazy_import("json")

    class B(A):
        pass

    assert B.MOD is json
    assert vars(A)["MOD"] is json
    assert "MOD" not in vars(B)


def test_lazy_import_subclass_override_is_kept() -> None:
    import json

    class A:
        MOD = lazy_import("json")

    class B(A):
        MOD = None

    assert B.MOD is None
    assert A.MOD is json
    assert B.MOD is None


def test_lazy_import_through_super() -> None:
    import json

    class A:
        MOD = lazy_import("json")

    class B(A):
        @property
        def MOD(self) -> tx.Any:  # noqa: N802
            return ("B", super().MOD)

    assert B().MOD == ("B", json)
    assert vars(A)["MOD"] is json


def test_lazy_import_bound_to_two_names() -> None:
    import json

    class A:
        ONE = TWO = lazy_import("json")

    assert A.ONE is json
    assert vars(A)["ONE"] is json and vars(A)["TWO"] is json


def test_lazy_import_keeps_descriptor_values_off_the_class() -> None:
    """A function value would bind as a method: it is not put on the class."""
    import json

    class A:
        DUMPS = lazy_import("json:dumps")

    assert A().DUMPS is json.dumps
    assert isinstance(vars(A)["DUMPS"], lazy_import)
    assert A().DUMPS is json.dumps


def test_lazy_import_without_set_name() -> None:
    import json

    class A:
        pass

    A.MOD = lazy_import("json")  # type: ignore[attr-defined]
    assert A.MOD is json  # type: ignore[attr-defined]
    assert A.MOD is json  # type: ignore[attr-defined]


def test_lazy_import_concurrent_first_access(pkg: str) -> None:
    class A:
        X = lazy_import(f"{pkg}.sub:Outer")

    results: tx.List[tx.Any] = []
    barrier = threading.Barrier(8)

    def read() -> None:
        barrier.wait()
        results.append(A.X)

    threads = [threading.Thread(target=read) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 8
    assert all(r is results[0] for r in results)
    assert vars(A)["X"] is results[0]


# --- when pending keys are retried ------------------------------------


def _count_finds(monkeypatch: tx.Any) -> tx.List[str]:
    calls: tx.List[str] = []
    find = _lazy._find

    def counting(name: str, context: tx.Any) -> tx.Any:
        calls.append(name)
        return find(name, context)

    monkeypatch.setattr(_lazy, "_find", counting)
    return calls


def test_steady_state_lookups_do_not_rescan(monkeypatch: tx.Any) -> None:
    registry: tx.Dict[tx.Any, tx.Any] = {}
    _module("_bagof_lazy_never", monkeypatch)  # finished, has no `Thing`
    _lazy.defer(
        registry, tx.ForwardRef("_bagof_lazy_never.Thing"), 1, __name__
    )
    _lazy.resolve_pending()
    calls = _count_finds(monkeypatch)
    for _ in range(3):
        _lazy.resolve_pending()
    assert calls == []
    assert _lazy.pending(registry) == [
        (__name__, "_bagof_lazy_never.Thing", 1)
    ]


def test_new_module_triggers_a_retry(monkeypatch: tx.Any) -> None:
    """A cross-package re-export resolves once its module is imported."""
    registry: tx.Dict[tx.Any, tx.Any] = {}
    _lazy.defer(registry, tx.ForwardRef("_bagof_lazy_new.Thing"), 1, "x")
    _lazy.resolve_pending()
    assert registry == {}
    # `int` lives in another package (a re-export): only the size changed
    _module("_bagof_lazy_new", monkeypatch).Thing = int
    _lazy.resolve_pending()
    assert registry == {int: 1}


def test_initialising_module_is_retried(monkeypatch: tx.Any) -> None:
    """A module still executing may grow without `sys.modules` changing."""
    registry: tx.Dict[tx.Any, tx.Any] = {}
    module = _module("_bagof_lazy_init", monkeypatch)
    module.__spec__._initializing = True
    _lazy.defer(registry, tx.ForwardRef("_bagof_lazy_init.Thing"), 1, "x")
    _lazy.resolve_pending()
    assert _lazy._DIRTY
    module.Thing = int
    module.__spec__._initializing = False
    _lazy.resolve_pending()
    assert registry == {int: 1}


def test_finished_module_is_settled(monkeypatch: tx.Any) -> None:
    registry: tx.Dict[tx.Any, tx.Any] = {}
    _module("_bagof_lazy_done", monkeypatch)
    _lazy.defer(registry, tx.ForwardRef("_bagof_lazy_done.Thing"), 1, "x")
    _lazy.resolve_pending()
    assert not _lazy._DIRTY


def test_force_picks_up_a_late_binding(monkeypatch: tx.Any) -> None:
    """A finished module patched later is only seen by a forced retry."""
    registry: tx.Dict[tx.Any, tx.Any] = {}
    module = _module("_bagof_lazy_late", monkeypatch)
    _lazy.defer(registry, tx.ForwardRef("_bagof_lazy_late.Thing"), 1, "x")
    _lazy.resolve_pending()
    module.Thing = int  # no import, no size change
    _lazy.resolve_pending()
    assert registry == {}
    _lazy.resolve_pending(force=True)
    assert registry == {int: 1}

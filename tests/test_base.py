"""Tests for the base converter machinery."""

# stdlib
import sys

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.converters.base import Converter


def test_register_accepts_several_hints() -> None:
    """`register` takes any number of hints, as its docstring says."""
    # The overloads annotated `*hints` as `Unpack[Tuple[Any]]` -- a
    # *one*-element tuple, so "exactly one hint" -- while the
    # implementation and the docstring both say "one or more".
    registry: tx.Dict[tx.Any, tx.Any] = {}

    class _A:
        pass

    class _B:
        pass

    class Multi(Converter):
        DEFAULT = _A

    Converter.register(Multi, _A, _B, registry=registry)
    assert registry[_A] is Multi
    assert registry[_B] is Multi


def test_register_true_registers_the_default_hint() -> None:
    """`register=True` in the class kwargs registers `DEFAULT`."""
    class _Registered:
        pass

    class ForDefault(Converter, register=True):
        DEFAULT = _Registered

    assert Converter.get_class(_Registered) is ForDefault


def test_register_a_single_hint_without_a_tuple() -> None:
    """`register=<hint>` is accepted as well as `register=(<hint>,)`."""
    class _Single:
        pass

    class ForSingle(Converter, register=_Single):
        DEFAULT = _Single

    assert Converter.get_class(_Single) is ForSingle


def test_error_defaults_to_a_generic_message() -> None:
    from bagof.converters.exceptions import (
        ConversionError,
        TypeConversionError,
        ValueConversionError,
    )

    converter = Converter(int)

    error = converter.error(object())
    assert isinstance(error, ConversionError)
    assert "Invalid value." in str(error)

    type_error = converter.type_error(object())
    assert isinstance(type_error, TypeConversionError)
    assert "Invalid value type" in str(type_error)

    value_error = converter.value_error(object())
    assert isinstance(value_error, ValueConversionError)
    assert "Invalid value." in str(value_error)


def test_error_maps_the_short_type_names() -> None:
    from bagof.converters.exceptions import (
        TypeConversionError,
        ValueConversionError,
    )

    converter = Converter(int)
    assert isinstance(converter.error(1, type="value"), ValueConversionError)
    assert isinstance(converter.error(1, type="type"), TypeConversionError)


def test_conversion_error_accepts_the_converter_alias() -> None:
    """`converter=` is accepted as a spelling of `this=`."""
    from bagof.converters.exceptions import ConversionError

    converter = Converter(int)
    error = ConversionError("boom", converter=converter)
    assert error.this is converter



# --- lazy forward-reference keys (#55) --------------------------------


@pytest.fixture(autouse=True)
def _restore_pending() -> tx.Iterator[None]:
    """Drop the pending entries a test leaves behind."""
    from bagof.converters import _lazy

    saved = list(_lazy._PENDING)
    yield
    _lazy._PENDING[:] = saved


def _fake_module(name: str, monkeypatch: tx.Any) -> tx.Any:
    import sys
    import types

    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _Thing:
    pass


def test_forward_ref_key_waits_for_its_module(monkeypatch: tx.Any) -> None:
    """A `ForwardRef` key is registered once its module is imported."""
    import sys

    registry: tx.Dict[tx.Any, tx.Any] = {}

    class ToLazy(Converter):
        pass

    Converter.register(
        ToLazy, tx.ForwardRef("_bagof_lazy_mod.Thing"), registry=registry
    )
    assert "_bagof_lazy_mod" not in sys.modules
    assert registry == {}

    _fake_module("_bagof_lazy_mod", monkeypatch).Thing = _Thing
    assert Converter.get_class(_Thing, registry) is ToLazy
    assert registry == {_Thing: ToLazy}


def test_forward_ref_key_of_a_partial_module_stays_pending(
    monkeypatch: tx.Any,
) -> None:
    """A module still initialising (name not defined yet) is retried."""
    registry: tx.Dict[tx.Any, tx.Any] = {}

    class ToLazy(Converter):
        pass

    module = _fake_module("_bagof_lazy_partial", monkeypatch)
    Converter.register(
        ToLazy, tx.ForwardRef("_bagof_lazy_partial.Thing"), registry=registry
    )
    assert registry == {}

    module.Thing = _Thing
    assert Converter.get_class(_Thing, registry) is ToLazy


def test_forward_ref_key_to_a_nested_class(monkeypatch: tx.Any) -> None:
    """The part after the module is walked as attributes."""
    registry: tx.Dict[tx.Any, tx.Any] = {}

    class Outer:
        class Inner:
            pass

    class ToLazy(Converter):
        pass

    _fake_module("_bagof_lazy_nested", monkeypatch).Outer = Outer
    Converter.register(
        ToLazy,
        tx.ForwardRef("_bagof_lazy_nested.Outer.Inner"),
        registry=registry,
    )
    assert registry == {Outer.Inner: ToLazy}


def test_forward_ref_key_of_an_unloaded_submodule_stays_pending(
    monkeypatch: tx.Any,
) -> None:
    """A loaded parent package does not resolve an unloaded submodule."""
    registry: tx.Dict[tx.Any, tx.Any] = {}

    class ToLazy(Converter):
        pass

    _fake_module("_bagof_lazy_pkg", monkeypatch)
    Converter.register(
        ToLazy, tx.ForwardRef("_bagof_lazy_pkg.sub.Thing"), registry=registry
    )
    assert Converter.get_class(_Thing, registry, fallback=None) is None
    assert registry == {}

    # the longest loaded prefix is used once the submodule is imported
    _fake_module("_bagof_lazy_pkg.sub", monkeypatch).Thing = _Thing
    assert Converter.get_class(_Thing, registry) is ToLazy


def test_plain_string_key_is_not_lazy() -> None:
    """Only `ForwardRef` keys are lazy; a string is an ordinary key."""
    registry: tx.Dict[tx.Any, tx.Any] = {}

    class ToStr(Converter):
        pass

    Converter.register(ToStr, "a:b", registry=registry)
    assert registry == {"a:b": ToStr}


def test_forward_ref_key_and_real_key_last_registration_wins(
    monkeypatch: tx.Any,
) -> None:
    """Forward-ref and real keys for one object keep "last one wins"."""
    registry: tx.Dict[tx.Any, tx.Any] = {}

    class First(Converter):
        pass

    class Second(Converter):
        pass

    class Third(Converter):
        pass

    # ref, then ref: the second replaces the first while pending (the
    # refs compare by name, whatever else `ForwardRef` equality involves)
    ref = "_bagof_lazy_order.Thing"
    Converter.register(First, tx.ForwardRef(ref), registry=registry)
    Converter.register(Second, tx.ForwardRef(ref), registry=registry)
    module = _fake_module("_bagof_lazy_order", monkeypatch)
    module.Thing = _Thing
    # real key registered after: it wins over the pending ref key
    Converter.register(Third, _Thing, registry=registry)
    assert Converter.get_class(_Thing, registry) is Third

    # real, then ref (module loaded): the ref key wins
    Converter.register(First, tx.ForwardRef(ref), registry=registry)
    assert Converter.get_class(_Thing, registry) is First


def _converter_in(module: str) -> tx.Any:
    """A fresh converter class whose `__module__` is `module`."""
    return type(Converter)("ToLazy", (Converter,), {"__module__": module})


def test_forward_ref_key_relative_to_the_registering_module(
    monkeypatch: tx.Any,
) -> None:
    """A bare name is looked up in the registering class's module."""
    registry: tx.Dict[tx.Any, tx.Any] = {}
    module = _fake_module("_bagof_lazy_rel", monkeypatch)
    ToLazy = _converter_in("_bagof_lazy_rel")

    Converter.register(ToLazy, tx.ForwardRef("MyThing"), registry=registry)
    assert registry == {}
    # defined later in the module (e.g. after the converter class)
    module.MyThing = _Thing
    assert Converter.get_class(_Thing, registry) is ToLazy


def test_forward_ref_key_relative_alias_of_a_runtime_import(
    monkeypatch: tx.Any,
) -> None:
    """`"np.Thing"` follows the module's own runtime `import ... as np`."""
    registry: tx.Dict[tx.Any, tx.Any] = {}
    target = _fake_module("_bagof_lazy_target", monkeypatch)
    target.Thing = _Thing
    _fake_module("_bagof_lazy_alias", monkeypatch).np = target
    ToLazy = _converter_in("_bagof_lazy_alias")

    Converter.register(ToLazy, tx.ForwardRef("np.Thing"), registry=registry)
    assert registry == {_Thing: ToLazy}


@pytest.mark.skipif(
    sys.version_info < (3, 9, 7), reason="ForwardRef(module=) is 3.9.7+"
)
def test_forward_ref_key_honours_module(monkeypatch: tx.Any) -> None:
    """`ForwardRef(name, module=...)` is relative to that module."""
    registry: tx.Dict[tx.Any, tx.Any] = {}
    ToLazy = _converter_in("_bagof_lazy_elsewhere")

    ref = tx.ForwardRef("Thing", module="_bagof_lazy_modarg")
    Converter.register(ToLazy, ref, registry=registry)
    assert registry == {}
    _fake_module("_bagof_lazy_modarg", monkeypatch).Thing = _Thing
    assert Converter.get_class(_Thing, registry) is ToLazy


def test_forward_ref_key_type_checking_alias_stays_pending(
    monkeypatch: tx.Any,
) -> None:
    """An alias that only exists for type checkers cannot be resolved."""
    from bagof.converters import _lazy

    registry: tx.Dict[tx.Any, tx.Any] = {}
    _fake_module("_bagof_lazy_tc", monkeypatch)  # no runtime `xx` alias
    _fake_module("_bagof_lazy_real", monkeypatch).Thing = _Thing
    ToLazy = _converter_in("_bagof_lazy_tc")

    Converter.register(ToLazy, tx.ForwardRef("xx.Thing"), registry=registry)
    assert registry == {}
    assert _lazy.pending(registry) == [("_bagof_lazy_tc", "xx.Thing", ToLazy)]

    # the absolute dotted path resolves
    Converter.register(
        ToLazy, tx.ForwardRef("_bagof_lazy_real.Thing"), registry=registry
    )
    assert registry == {_Thing: ToLazy}


def test_forward_ref_key_relative_wins_over_absolute(
    monkeypatch: tx.Any,
) -> None:
    """A module global shadows a top-level module of the same name."""
    registry: tx.Dict[tx.Any, tx.Any] = {}

    class Other:
        pass

    absolute = _fake_module("_bagof_lazy_clash", monkeypatch)
    absolute.Thing = Other
    local = _fake_module("_bagof_lazy_clash_ctx", monkeypatch)
    relative = _fake_module("_bagof_lazy_clash_rel", monkeypatch)
    relative.Thing = _Thing
    local._bagof_lazy_clash = relative
    ToLazy = _converter_in("_bagof_lazy_clash_ctx")

    Converter.register(
        ToLazy, tx.ForwardRef("_bagof_lazy_clash.Thing"), registry=registry
    )
    assert registry == {_Thing: ToLazy}


def test_forward_ref_key_relative_miss_falls_back_to_absolute(
    monkeypatch: tx.Any,
) -> None:
    """A relative first part whose walk fails still tries the absolute."""
    registry: tx.Dict[tx.Any, tx.Any] = {}
    _fake_module("_bagof_lazy_fb", monkeypatch).Thing = _Thing
    # the context has a `_bagof_lazy_fb` global without `Thing`
    _fake_module("_bagof_lazy_fb_ctx", monkeypatch)._bagof_lazy_fb = object()
    ToLazy = _converter_in("_bagof_lazy_fb_ctx")

    Converter.register(
        ToLazy, tx.ForwardRef("_bagof_lazy_fb.Thing"), registry=registry
    )
    assert registry == {_Thing: ToLazy}

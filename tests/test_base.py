"""Tests for the base converter machinery."""

# dependencies
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


# --- lazy name keys (#55) ---------------------------------------------


def _fake_module(name: str, monkeypatch: tx.Any) -> tx.Any:
    import sys
    import types

    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_name_key_waits_for_its_module(monkeypatch: tx.Any) -> None:
    """A `"module:name"` key is registered once the module is imported."""
    import sys

    registry: tx.Dict[tx.Any, tx.Any] = {}

    class ToLazy(Converter):
        pass

    Converter.register(ToLazy, "_bagof_lazy_mod:Thing", registry=registry)
    assert "_bagof_lazy_mod" not in sys.modules
    assert registry == {}

    class Thing:
        pass

    _fake_module("_bagof_lazy_mod", monkeypatch).Thing = Thing
    assert Converter.get_class(Thing, registry) is ToLazy
    assert registry == {Thing: ToLazy}


def test_name_key_of_a_partial_module_stays_pending(
    monkeypatch: tx.Any,
) -> None:
    """A module still initialising (name not defined yet) is retried."""
    registry: tx.Dict[tx.Any, tx.Any] = {}

    class ToLazy(Converter):
        pass

    module = _fake_module("_bagof_lazy_partial", monkeypatch)
    Converter.register(
        ToLazy, "_bagof_lazy_partial:Thing", registry=registry
    )
    assert registry == {}

    class Thing:
        pass

    module.Thing = Thing
    assert Converter.get_class(Thing, registry) is ToLazy


def test_name_key_and_real_key_last_registration_wins(
    monkeypatch: tx.Any,
) -> None:
    """Name and real keys for one object keep "last one wins"."""
    registry: tx.Dict[tx.Any, tx.Any] = {}

    class Thing:
        pass

    class First(Converter):
        pass

    class Second(Converter):
        pass

    class Third(Converter):
        pass

    # name, then name: the second replaces the first while pending
    Converter.register(First, "_bagof_lazy_order:Thing", registry=registry)
    Converter.register(Second, "_bagof_lazy_order:Thing", registry=registry)
    module = _fake_module("_bagof_lazy_order", monkeypatch)
    module.Thing = Thing
    # real key registered after: it wins over the pending name key
    Converter.register(Third, Thing, registry=registry)
    assert Converter.get_class(Thing, registry) is Third

    # real, then name (module loaded): the name key wins
    Converter.register(First, "_bagof_lazy_order:Thing", registry=registry)
    assert Converter.get_class(Thing, registry) is First

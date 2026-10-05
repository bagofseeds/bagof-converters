# stdlib
import enum

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.converters import enums
from bagof.converters.base import Converter
from bagof.converters.exceptions import ConversionError


class Colour(enum.Enum):
    RED = 1
    GREEN = 2


class Size(enum.IntEnum):
    SMALL = 1
    LARGE = 2


class Tag(enum.Flag):
    A = 1
    B = 2


@pytest.mark.parametrize(
    "value,expected",
    [
        # by value
        (1, Colour.RED),
        (2, Colour.GREEN),
        # by name
        ("RED", Colour.RED),
        ("GREEN", Colour.GREEN),
        # a member passes through
        (Colour.RED, Colour.RED),
    ],
)
def test_enum_valid(value: tx.Any, expected: Colour) -> None:
    assert enums.ToEnum(Colour)(value) is expected


@pytest.mark.parametrize("value", [3, "BLUE", None, 1.5])
def test_enum_invalid(value: tx.Any) -> None:
    with pytest.raises(ConversionError):
        enums.ToEnum(Colour)(value)


def test_enum_value_takes_precedence_over_name() -> None:
    # A member whose *name* is also a valid *value* resolves by value first.
    class Weird(enum.Enum):
        A = "B"
        B = "C"

    assert enums.ToEnum(Weird)("B") is Weird.A


@pytest.mark.parametrize(
    "cls,value,expected",
    [
        # subclasses dispatch to the enum.Enum registration
        (Size, 1, Size.SMALL),
        (Size, "LARGE", Size.LARGE),
        (Tag, 1, Tag.A),
        (Tag, "B", Tag.B),
    ],
)
def test_enum_subclasses_dispatch(
    cls: tx.Any, value: tx.Any, expected: tx.Any
) -> None:
    assert Converter.get(cls)(value) is expected


@pytest.mark.parametrize("cls", [Colour, Size, Tag, enum.Enum])
def test_enum_is_registered(cls: tx.Any) -> None:
    assert Converter.get_class(cls) is enums.ToEnum


def test_int_enum_accepts_bool_by_value() -> None:
    # ``True == 1`` so it resolves to the value-1 member.
    assert Converter.get(Size)(True) is Size.SMALL


# --- dispatch over mixed-in bases (``refines=True``) ------------------


class StrTag(str, enum.Enum):
    A = "a"
    B = "b"


def test_str_enum_dispatches_to_to_enum() -> None:
    assert Converter.get_class(StrTag) is enums.ToEnum
    assert Converter.get(StrTag)("a") is StrTag.A
    # name lookup is the ToEnum behaviour that ToString lacked
    assert Converter.get(StrTag)("A") is StrTag.A


@pytest.mark.skipif(
    not hasattr(enum, "StrEnum"), reason="enum.StrEnum needs Python 3.11+"
)
def test_stdlib_str_enum_dispatches_to_to_enum() -> None:
    class S(enum.StrEnum):  # type: ignore[name-defined,misc]
        A = "a"

    assert Converter.get_class(S) is enums.ToEnum
    assert Converter.get(S)("A") is S.A


class _Custom(Converter[tx.Any, tx.Any]):
    pass


def _registry() -> tx.Any:
    registry: tx.Any = {str: _Custom, int: _Custom, enum.Enum: enums.ToEnum}
    return registry


def test_exact_key_beats_refining_base() -> None:
    registry = _registry()
    registry[StrTag] = _Custom
    assert Converter.get_class(StrTag, registry) is _Custom


def test_nearer_subclass_of_refining_base_wins() -> None:
    class MyBase(enum.Enum):
        pass

    class Sub(str, MyBase):
        A = "a"

    registry = _registry()
    registry[MyBase] = _Custom  # non-refining, but more specific than Enum
    assert Converter.get_class(Sub, registry) is _Custom


def test_nearer_unrelated_base_loses_to_refining_base() -> None:
    assert Converter.get_class(StrTag, _registry()) is enums.ToEnum


def test_dispatch_unchanged_without_refining_base() -> None:
    from bagof.converters import get_converter_class

    assert get_converter_class(bool) is not enums.ToEnum
    assert Converter.get_class(bool, _registry()) is _Custom
    assert Converter.get_class(str, _registry()) is _Custom

    class Plain:
        pass

    assert Converter.get_class(Plain, _registry(), fallback=None) is None
    assert Converter.get_class(Plain) is Converter

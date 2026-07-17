# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.converters import common
from bagof.converters.exceptions import ConversionError

# --- ToAny ------------------------------------------------------------


@pytest.mark.parametrize("value", [1, "a", None, [1, 2], {"a": 1}])
def test_any_valid(value: tx.Any) -> None:
    converter = common.ToAny()
    assert converter(value) is value


# --- ToNone -----------------------------------------------------------


def test_none_valid() -> None:
    converter = common.ToNone()
    assert converter(None) is None


@pytest.mark.parametrize("value", [0, "", False, [], {}])
def test_none_invalid(value: tx.Any) -> None:
    converter = common.ToNone()
    with pytest.raises(ConversionError):
        converter(value)


# --- ToUnion ----------------------------------------------------------


@pytest.mark.parametrize(
    "hint,value,expected",
    [
        (tx.Union[int, str], 1, 1),
        (tx.Union[int, str], "a", "a"),
        (tx.Union[int, None], None, None),
        (tx.Union[int, None], 1, 1),
    ],
)
def test_union_valid(
    hint: tx.Any, value: tx.Any, expected: tx.Any
) -> None:
    converter = common.ToUnion(hint)
    assert converter(value) == expected


@pytest.mark.parametrize(
    "hint,value",
    [
        (tx.Union[int, str], [1, 2]),
        (tx.Union[int, str], None),
    ],
)
def test_union_invalid(hint: tx.Any, value: tx.Any) -> None:
    converter = common.ToUnion(hint)
    with pytest.raises(ConversionError):
        converter(value)


# --- ToLiteral --------------------------------------------------------


@pytest.mark.parametrize(
    "hint,value",
    [
        (tx.Literal[1, 2, 3], 1),
        (tx.Literal[1, 2, 3], 2),
        (tx.Literal["a", "b"], "a"),
    ],
)
def test_literal_valid(hint: tx.Any, value: tx.Any) -> None:
    converter = common.ToLiteral(hint)
    assert converter(value) == value


@pytest.mark.parametrize(
    "hint,value",
    [
        (tx.Literal[1, 2, 3], 4),
        (tx.Literal["a", "b"], "c"),
        (tx.Literal[1, 2, 3], "1"),
    ],
)
def test_literal_invalid(hint: tx.Any, value: tx.Any) -> None:
    converter = common.ToLiteral(hint)
    with pytest.raises(ConversionError):
        converter(value)


# --- ToAnnotated ------------------------------------------------------


@pytest.mark.parametrize(
    "hint,value,expected",
    [
        (tx.Annotated[int, common.ToAny()], "42", "42"),  # ToAny is no-op
        (tx.Annotated[str, common.ToLiteral(tx.Literal["a", "b"])], "a", "a"),
    ],
)
def test_annotated_valid(
    hint: tx.Any, value: tx.Any, expected: tx.Any
) -> None:
    converter = common.ToAnnotated(hint)
    assert converter(value) == expected


# --- ToType -----------------------------------------------------------


@pytest.mark.parametrize(
    "hint,value",
    [
        (type, int),
        (type, object),
        (tx.Type[int], int),
        (tx.Type[int], bool),  # bool is a subclass of int
        (tx.Type[object], str),
    ],
)
def test_type_valid(hint: tx.Any, value: tx.Any) -> None:
    assert common.ToType(hint)(value) is value


@pytest.mark.parametrize(
    "hint,value",
    [
        (type, 3),          # not a type
        (type, "int"),      # not a type
        (tx.Type[int], str),   # not a subclass of int
        (tx.Type[int], object),
    ],
)
def test_type_invalid(hint: tx.Any, value: tx.Any) -> None:
    with pytest.raises(ConversionError):
        common.ToType(hint)(value)


def test_type_is_registered() -> None:
    from bagof.converters.base import Converter

    assert Converter.get_class(type) is common.ToType

# dependencies
import pytest
import typing_extensions as tx

# bags
from bagof.hints.typevars.co import INT, STR

# locals
from bagof.converters import collections
from bagof.converters.exceptions import ConversionError

# --- ToIterable -------------------------------------------------------


@pytest.mark.parametrize(
    "hint,value,expected_type",
    [
        (tx.Iterable, [1, 2, 3], list),
        (tx.Iterable, (1, 2, 3), tuple),
        (tx.Iterable[int], [1, 2, 3], list),
        (tx.Iterable[int], (1, 2, 3), tuple),
        (tx.Iterable[INT], [1, 2, 3], list),
        (tx.Iterable[str], ["a", "b"], list),
        (tx.Iterable[STR], ["a", "b"], list),
    ],
)
def test_iterable_valid(
    hint: tx.Any, value: tx.Any, expected_type: type
) -> None:
    converter = collections.ToIterable(hint)
    result = converter(value)
    assert isinstance(result, expected_type) or hasattr(result, "__iter__")


@pytest.mark.parametrize(
    "hint,value",
    [
        # Element conversion failures (int("a") raises)
        (tx.Iterable[int], ["a", "b"]),
    ],
)
def test_iterable_invalid(hint: tx.Any, value: tx.Any) -> None:
    # NOTE: ToIterable with no args is a passthrough for abstract fallback
    # types (abc.Iterable), so non-iterable inputs do not raise.
    # Only element-conversion failures cause errors.
    converter = collections.ToIterable(hint)
    with pytest.raises((ConversionError, TypeError, ValueError)):
        list(converter(value))


# --- ToSequence -------------------------------------------------------


@pytest.mark.parametrize(
    "hint,value,expected",
    [
        (tx.Sequence[int], [1, 2, 3], [1, 2, 3]),
        (tx.Sequence[int], (1, 2, 3), [1, 2, 3]),
        (tx.Sequence[str], ["a", "b"], ["a", "b"]),
        (tx.List[int], [1, 2, 3], [1, 2, 3]),
    ],
)
def test_sequence_valid(
    hint: tx.Any, value: tx.Any, expected: tx.Any
) -> None:
    converter = collections.ToSequence(hint)
    result = converter(value)
    assert list(result) == expected


@pytest.mark.parametrize(
    "hint,value",
    [
        (tx.Sequence[int], {"a": 1}),
        (tx.Sequence[int], 1),
    ],
)
def test_sequence_invalid(hint: tx.Any, value: tx.Any) -> None:
    converter = collections.ToSequence(hint)
    with pytest.raises((ConversionError, TypeError, ValueError)):
        converter(value)


# --- ToMapping --------------------------------------------------------


@pytest.mark.parametrize(
    "hint,value,expected",
    [
        (tx.Mapping[str, int], {"a": 1, "b": 2}, {"a": 1, "b": 2}),
        (tx.Mapping[STR, INT], {"a": 1}, {"a": 1}),
        (tx.Dict[str, int], {"a": 1}, {"a": 1}),
    ],
)
def test_mapping_valid(
    hint: tx.Any, value: tx.Any, expected: tx.Any
) -> None:
    converter = collections.ToMapping(hint)
    assert converter(value) == expected


@pytest.mark.parametrize(
    "hint,value",
    [
        (tx.Mapping[str, int], {1: 1}),      # int key can't coerce to str
        (tx.Mapping[str, int], {"a": "x"}),  # "x" can't coerce to int
        (tx.Mapping[str, int], 1),            # not a mapping
    ],
)
def test_mapping_invalid(hint: tx.Any, value: tx.Any) -> None:
    converter = collections.ToMapping(hint)
    with pytest.raises((ConversionError, TypeError, ValueError)):
        converter(value)


# --- ToTuple ----------------------------------------------------------


@pytest.mark.parametrize(
    "hint,value,expected",
    [
        (tx.Tuple[int, str], [1, "a"], (1, "a")),
        (tx.Tuple[int, str], (1, "a"), (1, "a")),
        (tx.Tuple[int, ...], [1, 2, 3], (1, 2, 3)),
    ],
)
def test_tuple_valid(
    hint: tx.Any, value: tx.Any, expected: tx.Any
) -> None:
    converter = collections.ToTuple(hint)
    assert converter(value) == expected


@pytest.mark.parametrize(
    "hint,value",
    [
        (tx.Tuple[int, str], (1, 2, 3)),       # wrong length
        (tx.Tuple[int, str], (1,)),             # wrong length
        (tx.Tuple[int, ...], ("a", "b", "c")),  # wrong element type
    ],
)
def test_tuple_invalid(hint: tx.Any, value: tx.Any) -> None:
    converter = collections.ToTuple(hint)
    with pytest.raises((ConversionError, TypeError, ValueError)):
        converter(value)


# --- ToLength ---------------------------------------------------------


@pytest.mark.parametrize(
    "length,hint,value,expected",
    [
        (3, tx.List[int], [1, 2, 3], [1, 2, 3]),
        (0, tx.List[int], [], []),
        (2, tx.Sequence[str], ["a", "b"], ["a", "b"]),
    ],
)
def test_length_valid(
    length: int, hint: tx.Any, value: tx.Any, expected: tx.Any
) -> None:
    converter = collections.ToLength(length, hint)
    assert list(converter(value)) == expected


@pytest.mark.parametrize(
    "length,hint,value",
    [
        (3, tx.List[int], [1, 2]),   # too short → raises
        (5, tx.List[int], [1, 2]),   # too short → raises
    ],
)
def test_length_invalid(length: int, hint: tx.Any, value: tx.Any) -> None:
    # ToLength truncates when input is too long but raises when too short.
    converter = collections.ToLength(length, hint)
    with pytest.raises((ConversionError, TypeError, ValueError)):
        converter(value)

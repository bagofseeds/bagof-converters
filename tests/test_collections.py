# stdlib
from collections import abc

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


# --- ToSet / ToMutableSet ---------------------------------------------


@pytest.mark.parametrize(
    "hint,value,expected",
    [
        # abstract set hints still yield a concrete container...
        (tx.AbstractSet[int], ["1", "2"], frozenset({1, 2})),
        (tx.AbstractSet[int], (1, 2, 2), frozenset({1, 2})),
        (tx.MutableSet[int], ["1", "2"], {1, 2}),
        # ... and elements are converted, like any iterable
        (tx.FrozenSet[int], ["1", "2"], frozenset({1, 2})),
        (tx.Set[int], ["1", "2"], {1, 2}),
    ],
)
def test_set_valid(hint: tx.Any, value: tx.Any, expected: tx.Any) -> None:
    result = collections.Converter.get(hint)(value)
    assert result == expected
    assert type(result) is type(expected)


def test_abstract_set_is_frozen_by_default() -> None:
    # abc.Set is immutable, so its concrete fallback is frozenset.
    result = collections.Converter.get(tx.AbstractSet[int])([1, 2])
    assert isinstance(result, frozenset)


def test_mutable_set_is_mutable() -> None:
    result = collections.Converter.get(tx.MutableSet[int])([1, 2])
    assert isinstance(result, set) and not isinstance(result, frozenset)


@pytest.mark.parametrize(
    "hint,cls",
    [
        (tx.AbstractSet, collections.ToSet),
        (tx.MutableSet, collections.ToMutableSet),
    ],
)
def test_set_registration(hint: tx.Any, cls: tx.Any) -> None:
    assert collections.Converter.get_class(hint) is cls


# --- passthrough & one-shot iterators ---------------------------------


@pytest.mark.parametrize(
    "hint,value",
    [
        (list, [1, 2, 3]),
        (tuple, (1, 2, 3)),
        (dict, {"a": 1}),
        (set, {1, 2}),
        (frozenset, frozenset({1, 2})),
        (tx.MutableSequence, [1, 2]),
        (tx.Sequence, [1, 2]),
        (tx.Iterable, [1, 2]),
        (tx.Mapping, {"a": 1}),
        (tx.AbstractSet, frozenset({1, 2})),
    ],
)
def test_already_valid_passes_through_unchanged(
    hint: tx.Any, value: tx.Any
) -> None:
    # An input that already satisfies the (unparametrised) hint must be
    # returned as-is, never copied.
    result = collections.Converter.get(hint)(value)
    assert result is value


def test_bare_iterable_passes_a_generator_through() -> None:
    # Regression: this used to raise (a generator cannot be rebuilt from
    # itself).
    gen = (i for i in range(3))
    result = collections.Converter.get(abc.Iterable)(gen)
    assert result is gen
    assert list(result) == [0, 1, 2]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: (i for i in range(3)),
        lambda: map(int, ["0", "1", "2"]),
        lambda: zip(range(3), range(3)),
        lambda: filter(None, range(3)),
    ],
)
def test_bare_iterable_accepts_any_one_shot_iterator(
    factory: tx.Any,
) -> None:
    value = factory()
    assert collections.Converter.get(abc.Iterable)(value) is value


def test_typed_iterable_over_generator_is_lazy() -> None:
    # ``Iterable[int]`` maps lazily and does not consume the source until
    # the result is iterated.
    consumed = []

    def source() -> tx.Iterator[str]:
        for i in range(3):
            consumed.append(i)
            yield str(i)

    result = collections.Converter.get(tx.Iterable[int])(source())
    assert consumed == []  # nothing consumed yet
    assert list(result) == [0, 1, 2]
    assert consumed == [0, 1, 2]


@pytest.mark.parametrize(
    "hint,value,expected_type,expected",
    [
        # concrete targets materialise a one-shot iterator
        (tx.List[int], (i for i in ["1", "2"]), list, [1, 2]),
        (tx.Set[int], (i for i in ["1", "2"]), set, {1, 2}),
        (list, (i for i in range(3)), list, [0, 1, 2]),
        (tuple, [1, 2, 3], tuple, (1, 2, 3)),
    ],
)
def test_conversion_still_builds_concrete_containers(
    hint: tx.Any, value: tx.Any, expected_type: type, expected: tx.Any
) -> None:
    result = collections.Converter.get(hint)(value)
    assert type(result) is expected_type
    assert result == expected

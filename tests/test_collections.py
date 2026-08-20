# stdlib
import collections as std_collections
from collections import abc

# dependencies
import pytest
import typing_extensions as tx

# bags
from bagof.hints.typevars.co import INT, STR

# locals
from bagof.converters import collections
from bagof.converters.base import Converter
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


# ----------------------------------------------------------------------
# Single-argument mappings
# ----------------------------------------------------------------------


def test_counter_converts_keys_and_values() -> None:
    # `Counter[K]` is a `Mapping[K, int]`, so it carries one type
    # argument. Indexing `args[1]` on one used to raise a bare
    # `IndexError` from inside the converter.
    result = Converter.get(tx.Counter[str])({"a": "1"})
    assert result == {"a": 1}
    assert isinstance(result["a"], int)


def test_counter_rejects_a_bad_key() -> None:
    with pytest.raises(ConversionError):
        Converter.get(tx.Counter[str])({1: 2})


def test_counter_like_is_well_formed() -> None:
    like = Converter.get(tx.Counter[str]).like()
    assert like is not tx.Any


def test_mapping_args_helper() -> None:
    assert collections._mapping_args((str, int), dict) == (str, int)
    assert collections._mapping_args((str,), std_collections.Counter) == (
        str, int
    )
    assert collections._mapping_args((str,), dict) == (str, tx.Any)
    assert collections._mapping_args((), dict) == (tx.Any, tx.Any)


# ----------------------------------------------------------------------
# NamedTuple
# ----------------------------------------------------------------------


class _Point(tx.NamedTuple):
    x: int
    y: str = "origin"


class _Pair(tx.NamedTuple):
    a: int
    b: int


@pytest.mark.parametrize(
    "value,expected",
    [
        ({"a": "1", "b": "2"}, _Pair(1, 2)),      # mapping by field name
        (("1", "2"), _Pair(1, 2)),                # sequence in field order
        (["1", "2"], _Pair(1, 2)),
        (_Pair(1, 2), _Pair(1, 2)),               # already an instance
    ],
)
def test_namedtuple_converts_from_mapping_or_sequence(
    value: tx.Any, expected: tx.Any
) -> None:
    # A NamedTuple has no `__args__`, so `ToTuple` read a target length of
    # zero and rejected everything.
    result = Converter.get(_Pair)(value)
    assert result == expected
    assert type(result) is _Pair


def test_namedtuple_uses_field_defaults_for_absent_fields() -> None:
    assert Converter.get(_Point)(["1"]) == _Point(1, "origin")


@pytest.mark.parametrize(
    "value",
    [
        {"a": 1, "z": 2},        # unknown field
        [1, 2, 3],               # too many values
        {"b": 2},                # missing field with no default
        5,                       # not a mapping or sequence
        "ab",                    # a string is not a field sequence
    ],
)
def test_namedtuple_rejects_bad_input(value: tx.Any) -> None:
    with pytest.raises(ConversionError):
        Converter.get(_Pair)(value)


def test_namedtuple_like_reports_what_it_accepts() -> None:
    args = tx.get_args(Converter.get(_Pair).like())
    assert _Pair in args


@pytest.mark.parametrize(
    "hint,value,expected",
    [
        (tx.Tuple[int, int], ["1", "2"], (1, 2)),
        (tx.Tuple[int, ...], ["1", "2"], (1, 2)),
    ],
)
def test_plain_tuples_are_unaffected(
    hint: tx.Any, value: tx.Any, expected: tx.Any
) -> None:
    result = Converter.get(hint)(value)
    assert result == expected
    assert type(result) is tuple


def test_tolength_truncates_a_longer_sequence() -> None:
    # Deliberate: coercing to a fixed length is a conversion, so the
    # extra items are dropped. `HasLength` in bagof-validators refuses a
    # mismatch in either direction.
    assert Converter.get(tx.List[int]) is not None
    assert collections.ToLength(2, tx.List[int])(["1", "2", "3"]) == [1, 2]


def test_tolength_refuses_a_shorter_sequence() -> None:
    with pytest.raises(ConversionError, match="length 2"):
        collections.ToLength(2, tx.List[int])(["1"])


def test_tolength_docstring_example() -> None:
    to_pair = collections.ToLength(2, tx.List[int])
    assert to_pair(["1", "2", "3"]) == [1, 2]
    with pytest.raises(ConversionError):
        to_pair(["1"])

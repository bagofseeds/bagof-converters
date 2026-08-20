"""Tests for `like()` and `wrap_converter`.

`like()` reports the hint describing *valid inputs* for a converter, and
`wrap_converter` turns that into a real annotation. Neither had tests,
which is how the `Optional` collapse (#20) went unnoticed.
"""

# stdlib
import numbers
from collections import abc

# dependencies
import pytest
import typing_extensions as tx

# bags
from bagof.core.magic import issubscriptable

# locals
from bagof.converters.base import Converter, wrap_converter
from bagof.converters.collections import _type_to_hint


def _members(hint: tx.Any) -> tx.Any:
    """The set of union members, flattened one level."""
    args = tx.get_args(hint)
    return set(args) if args else {hint}


# ----------------------------------------------------------------------
# Sequences and iterables
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "hint", [tx.List[int], tx.Sequence[int], tx.Set[int], tx.Iterable[int]]
)
def test_element_wise_like_is_an_iterable_of_the_element_like(
    hint: tx.Any,
) -> None:
    # The container accepts any iterable; the element hint is whatever
    # the element converter accepts.
    like = Converter.get(hint).like()
    assert tx.get_origin(like) in (abc.Iterable, tx.Iterable)
    (element,) = tx.get_args(like)
    assert numbers.Integral in _members(element)


def test_unparametrised_iterable_like_has_no_element_hint() -> None:
    assert Converter.get(abc.Iterable).like() in (tx.Iterable, abc.Iterable)


def test_sequence_like_reports_the_element_converter_input() -> None:
    like = Converter.get(tx.Sequence[str]).like()
    (element,) = tx.get_args(like)
    # `ToString` accepts bytes as well as str.
    assert {str, bytes} <= _members(element)


# ----------------------------------------------------------------------
# Mappings
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "hint", [tx.Dict[str, int], tx.Mapping[str, int], tx.Counter[str]]
)
def test_mapping_like_accepts_a_mapping_or_pairs(hint: tx.Any) -> None:
    # A mapping converter accepts either a mapping or an iterable of
    # key/value pairs, so `like()` is the union of the two.
    like = Converter.get(hint).like()
    origins = {tx.get_origin(member) for member in tx.get_args(like)}
    assert abc.Mapping in origins or tx.Mapping in origins
    assert abc.Iterable in origins or tx.Iterable in origins


def test_unparametrised_mapping_like_falls_back_to_any() -> None:
    like = Converter.get(abc.Mapping).like()
    assert tx.get_args(like)


# ----------------------------------------------------------------------
# Tuples
# ----------------------------------------------------------------------


def test_fixed_tuple_like_is_positional() -> None:
    like = Converter.get(tx.Tuple[int, str]).like()
    first, second = tx.get_args(like)
    assert numbers.Integral in _members(first)
    assert str in _members(second)


def test_variadic_tuple_like_is_an_iterable() -> None:
    # A trailing ellipsis does not line up positionally, so it reports
    # the iterable form instead.
    like = Converter.get(tx.Tuple[int, ...]).like()
    assert tx.get_origin(like) in (abc.Iterable, tx.Iterable)


def test_unparametrised_tuple_like() -> None:
    assert Converter.get(tuple).like() in (tx.Tuple, tuple)


# ----------------------------------------------------------------------
# _type_to_hint
# ----------------------------------------------------------------------


@pytest.mark.parametrize("value", [tx.List, tx.Dict, tx.List[int]])
def test_type_to_hint_keeps_a_subscriptable_hint(value: tx.Any) -> None:
    assert _type_to_hint(value) is value


@pytest.mark.parametrize("value,alias", [(list, tx.List), (dict, tx.Dict)])
def test_type_to_hint_maps_a_bare_builtin_to_its_typing_alias(
    value: tx.Any, alias: tx.Any
) -> None:
    # `list[int]` only became legal in 3.9; below that the builtin is
    # swapped for the `typing` alias, which is subscriptable everywhere.
    expected = value if issubscriptable(value) else alias
    assert _type_to_hint(value) is expected


def test_type_to_hint_leaves_an_unknown_type_alone() -> None:

    class Weird:
        pass

    assert _type_to_hint(Weird) is Weird


@pytest.mark.parametrize("value", [abc.Hashable, abc.Sized])
def test_type_to_hint_maps_a_bare_abc_to_its_typing_alias(
    value: tx.Any,
) -> None:
    # `Hashable` and `Sized` are not subscriptable, but `typing` has an
    # alias of the same name that is.
    if issubscriptable(value):
        pytest.skip("this runtime subscripts the abc directly")
    assert _type_to_hint(value) is getattr(tx, value.__name__)


# ----------------------------------------------------------------------
# wrap_converter
# ----------------------------------------------------------------------


def test_wrap_converter_annotates_from_and_to() -> None:
    wrapped = wrap_converter(Converter.get(tx.List[int]))
    assert wrapped.__annotations__["return"] == tx.List[int]
    assert wrapped.__annotations__["value"] == Converter.get(
        tx.List[int]
    ).like()


def test_wrap_converter_still_converts() -> None:
    wrapped = wrap_converter(Converter.get(tx.List[int]))
    assert wrapped(["1", "2"]) == [1, 2]


def test_wrap_converter_explicit_to_and_from() -> None:
    wrapped = wrap_converter(Converter.get(int), TO=float, FROM=str)
    assert wrapped.__annotations__ == {"value": str, "return": float}
    assert wrapped("5") == 5


def test_wrap_converter_derives_from_from_a_different_to() -> None:
    # When `TO` differs from the converter's own hint, `FROM` comes from
    # the converter for `TO`.
    wrapped = wrap_converter(Converter.get(int), TO=tx.List[int])
    assert wrapped.__annotations__["value"] == Converter.get(
        tx.List[int]
    ).like()


# ----------------------------------------------------------------------
# Re-entrancy
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "hint",
    [
        tx.List[int],
        tx.Sequence[int],
        tx.Mapping[str, int],
        tx.Tuple[int, str],
        tx.Union[int, str],
        tx.Optional[int],
    ],
)
def test_like_stops_on_re_entry(hint: tx.Any) -> None:
    # The guard exists so a self-referential hint cannot recurse forever.
    # Visiting a hint that is already on the stack returns it unchanged.
    converter = Converter.get(hint)
    assert converter.like((converter.unwrapped,)) is converter.unwrapped


def test_typevar_like_follows_the_bound() -> None:
    T = tx.TypeVar("T", bound=int)
    like = Converter.get(T).like()
    assert numbers.Integral in _members(like)


def test_typevar_like_stops_on_re_entry() -> None:
    T = tx.TypeVar("T", bound=int)
    converter = Converter.get(T)
    assert converter.like((converter.hint,)) is converter.hint


def test_annotated_like_uses_the_first_converter() -> None:
    like = Converter.get(tx.Annotated[tx.List[int], "meta"]).like()
    assert tx.get_origin(like) in (abc.Iterable, tx.Iterable)


def test_annotated_like_stops_on_re_entry() -> None:
    hint = tx.Annotated[int, "meta"]
    converter = Converter.get(hint)
    assert converter.like((hint,)) is hint


def test_literal_like_is_the_literal_itself() -> None:
    hint = tx.Literal[1, 2]
    assert Converter.get(hint).like() == hint


def test_typeddict_like_accepts_any_mapping() -> None:

    class Film(tx.TypedDict):
        title: str

    like = Converter.get(Film).like()
    assert tx.get_origin(like) in (abc.Mapping, tx.Mapping)


def test_bare_union_like_is_unset_on_re_entry() -> None:
    # locals
    from bagof.core.magic import UNSET

    from bagof.converters.common import _like_union

    assert _like_union(tx.Union, (tx.Union,)) is UNSET


def test_union_of_only_any_reports_any() -> None:
    assert Converter.get(tx.Union[tx.Any, tx.Any]).like() is tx.Any


def test_like_union_rejects_a_non_union() -> None:
    # locals
    from bagof.converters.common import _like_union

    with pytest.raises(TypeError, match="not a Union"):
        _like_union(int)


def test_empty_enum_like() -> None:
    # stdlib
    import enum

    class Empty(enum.Enum):
        pass

    like = Converter.get(Empty).like()
    assert Empty in _members(like)


# ----------------------------------------------------------------------
# _like_iterable
# ----------------------------------------------------------------------


def test_iterable_like_of_a_variadic_tuple_uses_the_item_type() -> None:
    # `Tuple[int, ...]` is homogeneous: the ellipsis is not a member type.
    like = Converter.get(abc.Iterable).like()
    assert like is not None

    from bagof.converters.collections import _like_iterable

    result = _like_iterable(tx.Tuple[int, ...])
    assert tx.get_origin(result) in (abc.Iterable, tx.Iterable)
    members = _members(tx.get_args(result)[0])
    assert numbers.Integral in members


def test_iterable_like_of_a_mapping_yields_key_value_pairs() -> None:
    from bagof.converters.collections import _like_iterable

    result = _like_iterable(tx.Mapping[str, int])
    assert tx.get_origin(result) in (abc.Iterable, tx.Iterable)
    (item,) = tx.get_args(result)
    assert tx.get_origin(item) in (tuple, tx.Tuple)


def test_iterable_like_stops_on_re_entry() -> None:
    from bagof.converters.collections import _like_iterable

    hint = tx.List[int]
    assert _like_iterable(hint, (hint,)) is hint

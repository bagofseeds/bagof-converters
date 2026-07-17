"""
A typevar hint must convert exactly like the hint it is bound to.

Converters introspect `origin`/`args` to walk a hint's structure. Since
`MagicHint.UNWRAP` includes `TypeVar`, a typevar is resolved to its bound
(or default, or constraints) first, so that building a converter directly
with a typevar, building it with the bound, and dispatching through the
registry all agree.
"""

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.converters import collections, common, numbers, strings
from bagof.converters.base import Converter

BOUND_TO_LIST = tx.TypeVar("BOUND_TO_LIST", bound=tx.List[int])
BOUND_TO_DICT = tx.TypeVar("BOUND_TO_DICT", bound=tx.Dict[str, int])
BOUND_TO_TUPLE = tx.TypeVar("BOUND_TO_TUPLE", bound=tx.Tuple[int, int])
BOUND_TO_UNION = tx.TypeVar("BOUND_TO_UNION", bound=tx.Union[int, str])
BOUND_TO_INT = tx.TypeVar("BOUND_TO_INT", bound=int)
BOUND_TO_STR = tx.TypeVar("BOUND_TO_STR", bound=str)

# (converter, bound hint, typevar bound to it, input value)
EQUIVALENCES = [
    (collections.ToIterable, tx.List[int], BOUND_TO_LIST, ["1", "2"]),
    (collections.ToSequence, tx.List[int], BOUND_TO_LIST, ["1", "2"]),
    (collections.ToMapping, tx.Dict[str, int], BOUND_TO_DICT, {"a": "1"}),
    (collections.ToTuple, tx.Tuple[int, int], BOUND_TO_TUPLE, ("1", "2")),
    (common.ToUnion, tx.Union[int, str], BOUND_TO_UNION, "1"),
    (numbers.ToNumber, int, BOUND_TO_INT, "1"),
    (strings.ToString, str, BOUND_TO_STR, "abc"),
]

IDS = [case[0].__name__ for case in EQUIVALENCES]


@pytest.mark.parametrize("cls,hint,typevar,value", EQUIVALENCES, ids=IDS)
def test_typevar_converts_like_its_bound(
    cls: tx.Any, hint: tx.Any, typevar: tx.Any, value: tx.Any
) -> None:
    # Regression: `args` used to be empty for a typevar, so items were
    # silently left unconverted -- `ToSequence(TV)(["1"])` returned
    # `["1"]` (a list of `str`) instead of `[1]`.
    expected = cls(hint)(value)
    assert cls(typevar)(value) == expected
    assert Converter.get(typevar)(value) == expected


@pytest.mark.parametrize("cls,hint,typevar,value", EQUIVALENCES, ids=IDS)
def test_typevar_converts_to_the_same_types(
    cls: tx.Any, hint: tx.Any, typevar: tx.Any, value: tx.Any
) -> None:
    # Equality is not enough: `1 == True` and `["1"] != [1]`, but a
    # container's *item* types must match too.
    expected = cls(hint)(value)
    result = cls(typevar)(value)
    assert type(result) is type(expected)
    if isinstance(expected, dict):
        assert (
            [type(v) for v in result.values()]
            == [type(v) for v in expected.values()]
        )
    elif isinstance(expected, (list, tuple)):
        assert [type(v) for v in result] == [type(v) for v in expected]


@pytest.mark.parametrize("cls,hint,typevar,value", EQUIVALENCES, ids=IDS)
def test_typevar_like_matches_its_bound(
    cls: tx.Any, hint: tx.Any, typevar: tx.Any, value: tx.Any
) -> None:
    assert cls(typevar).like() == cls(hint).like()


@pytest.mark.parametrize("cls,hint,typevar,value", EQUIVALENCES, ids=IDS)
def test_typevar_introspection_matches_its_bound(
    cls: tx.Any, hint: tx.Any, typevar: tx.Any, value: tx.Any
) -> None:
    assert cls(typevar).origin == cls(hint).origin
    assert cls(typevar).args == cls(hint).args
    assert cls(typevar).unwrapped == cls(hint).unwrapped


def test_sequence_items_are_converted_through_a_typevar() -> None:
    # The regression, spelled out.
    assert collections.ToSequence(BOUND_TO_LIST)(["1", "2"]) == [1, 2]
    assert all(
        type(x) is int
        for x in collections.ToSequence(BOUND_TO_LIST)(["1", "2"])
    )


def test_mapping_values_are_converted_through_a_typevar() -> None:
    result = collections.ToMapping(BOUND_TO_DICT)({"a": "1"})
    assert result == {"a": 1}
    assert all(type(v) is int for v in result.values())


def test_constrained_typevar_converts_as_a_union() -> None:
    constrained = tx.TypeVar("CONSTRAINED", int, str)
    assert common.ToUnion(constrained)("1") == common.ToUnion(
        tx.Union[int, str]
    )("1")


def test_typevar_with_default() -> None:
    hint = tx.TypeVar("WITH_DEFAULT", default=tx.List[int])
    assert collections.ToSequence(hint)(["1", "2"]) == [1, 2]

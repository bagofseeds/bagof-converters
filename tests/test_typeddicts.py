"""Tests for the TypedDict converter."""

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.converters.base import Converter
from bagof.converters.exceptions import ConversionError
from bagof.converters.typeddicts import ToTypedDict, _strip_requiredness


class Movie(tx.TypedDict):
    title: str
    year: int


class Partial(tx.TypedDict):
    a: int
    b: tx.NotRequired[str]


class Wrapped(tx.TypedDict):
    # PEP 655 allows either nesting, and typing keeps whichever was
    # written, so `Annotated` on the outside hides the `NotRequired`.
    outside: tx.Annotated[tx.NotRequired[int], "meta"]
    inside: tx.NotRequired[tx.Annotated[int, "meta"]]
    plain: int


def test_typeddict_is_registered() -> None:
    assert isinstance(Converter.get(Movie), ToTypedDict)


def test_fields_are_converted() -> None:
    # A TypedDict used to land on `ToMapping`, which saw no type
    # arguments and returned the input untouched.
    assert Converter.get(Movie)({"title": "Alien", "year": "1979"}) == {
        "title": "Alien",
        "year": 1979,
    }


def test_converted_result_has_the_declared_field_types() -> None:
    # The whole point: the output must actually match the TypedDict, so
    # that validating it downstream succeeds.
    result = Converter.get(Movie)({"title": "a", "year": "1"})
    assert type(result["title"]) is str
    assert type(result["year"]) is int


def test_optional_key_may_be_absent() -> None:
    assert Converter.get(Partial)({"a": "1"}) == {"a": 1}


def test_optional_key_is_converted_when_present() -> None:
    assert Converter.get(Partial)({"a": "1", "b": b"x"}) == {"a": 1, "b": "x"}


def test_missing_required_key_raises() -> None:
    # Building a value for an absent key is a factory's job.
    with pytest.raises(ConversionError, match="Missing required key"):
        Converter.get(Movie)({"title": "x"})


def test_bad_field_raises_with_a_useful_message() -> None:
    with pytest.raises(ConversionError):
        Converter.get(Movie)({"title": "x", "year": "not a number"})


@pytest.mark.parametrize(
    "value,expected",
    [
        ({"plain": "1"}, {"plain": 1}),
        ({"plain": 1, "outside": "2"}, {"plain": 1, "outside": 2}),
        ({"plain": 1, "inside": "3"}, {"plain": 1, "inside": 3}),
    ],
)
def test_requiredness_under_annotated(
    value: tx.Any, expected: tx.Any
) -> None:
    assert Converter.get(Wrapped)(value) == expected


def test_extra_keys_are_kept_when_open() -> None:
    result = Converter.get(Movie)({"title": "x", "year": 1, "extra": 9})
    assert result["extra"] == 9


def test_non_mapping_input_raises() -> None:
    with pytest.raises(ConversionError):
        Converter.get(Movie)(5)


def test_inherited_totality() -> None:

    class Base(tx.TypedDict, total=False):
        optional_key: int

    class Child(Base):
        required_key: int

    assert Converter.get(Child)({"required_key": "1"}) == {"required_key": 1}
    with pytest.raises(ConversionError, match="required_key"):
        Converter.get(Child)({"optional_key": 1})


@pytest.mark.parametrize(
    "hint,expected",
    [
        (tx.NotRequired[int], int),
        (tx.Required[int], int),
        (tx.Annotated[tx.NotRequired[int], "m"], tx.Annotated[int, "m"]),
        (tx.NotRequired[tx.Annotated[int, "m"]], tx.Annotated[int, "m"]),
        (int, int),
    ],
)
def test_strip_requiredness(hint: tx.Any, expected: tx.Any) -> None:
    assert _strip_requiredness(hint) == expected

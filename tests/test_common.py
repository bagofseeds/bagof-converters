# stdlib
import enum

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.converters import common
from bagof.converters.base import Converter, wrap_converter
from bagof.converters.exceptions import ConversionError


class _Unregistered:
    """A type no converter is registered for."""


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


# --- ToUnion.like() -----------------------------------------------------
#
# `.like()` returns the hint describing valid inputs for a union, with
# redundant (already-covered) branches filtered out. These classes give a
# real, reliable `issubhint` relationship to exercise that filtering,
# without depending on any generic-parametrised-hint behaviour.


class _Narrow(int):
    """A strict subclass of `int` -- redundant once `int` is also present."""


class _SiblingA(int):
    """A strict subclass of `int`, unrelated to `_SiblingB`."""


class _SiblingB(int):
    """A strict subclass of `int`, unrelated to `_SiblingA`."""


def _like_int() -> tx.Any:
    """What `int` alone accepts as input.

    Not simply `int`: the numpy converter widens it when numpy is
    installed, so comparing against a literal type would pass or fail
    depending on which optional libraries are present.
    """
    return Converter.get(int).like()


def test_union_like_drops_redundant_subhint() -> None:
    # `_Narrow` is a subclass of `int`, so once `int` is also a branch,
    # listing `_Narrow` separately is redundant: the union must accept
    # exactly what `int` alone accepts.
    assert common.ToUnion(tx.Union[int, _Narrow]).like() == _like_int()


def test_union_like_is_order_independent() -> None:
    # The same redundant branch must be dropped regardless of the order
    # the union members appear in.
    forward = common.ToUnion(tx.Union[int, _Narrow]).like()
    backward = common.ToUnion(tx.Union[_Narrow, int]).like()
    assert forward == backward == _like_int()


def test_union_like_drops_multiple_superseded_subhints() -> None:
    # `_SiblingA` and `_SiblingB` are unrelated to each other, so both
    # survive on their own -- but `int` supersedes both at once and must
    # remove them both, not just the first match.
    union = tx.Union[_SiblingA, _SiblingB, int]
    assert common.ToUnion(union).like() == _like_int()


# ----------------------------------------------------------------------
# Lookup fallback
# ----------------------------------------------------------------------


def test_get_falls_back_to_the_base_converter() -> None:
    # `Validator.get` and `Factory.get` both default to their base class;
    # this one returned `None`, so an unregistered hint failed later and
    # elsewhere with "'NoneType' object is not callable".
    converter = Converter.get(_Unregistered)
    assert isinstance(converter, Converter)


def test_unregistered_hint_raises_a_conversion_error_naming_it() -> None:
    with pytest.raises(ConversionError) as info:
        Converter.get(_Unregistered)("nope")
    assert "_Unregistered" in str(info.value)


def test_base_converter_passes_an_already_valid_value_through() -> None:
    value = _Unregistered()
    assert Converter.get(_Unregistered)(value) is value


def test_get_still_returns_none_when_asked() -> None:
    assert Converter.get(_Unregistered, fallback=None) is None
    assert Converter.get_class(_Unregistered, fallback=None) is None


def test_wrap_converter_with_an_unregistered_target() -> None:
    # `Converter.get(TO)` returned `None` here, and `wrap_converter`
    # dereferenced it: "'NoneType' object has no attribute 'like'".
    wrapped = wrap_converter(Converter.get(int), _Unregistered)
    assert callable(wrapped)


# ----------------------------------------------------------------------
# like()
# ----------------------------------------------------------------------


def test_optional_like_does_not_collapse_to_any() -> None:
    # `ToNone` inherited the base's `Any`, and `Any` is a super hint of
    # every other branch -- so the redundancy filter erased them all and
    # the most common hint shape in Python reported no information.
    like = Converter.get(tx.Optional[int]).like()
    assert like is not tx.Any
    assert type(None) in tx.get_args(like)


@pytest.mark.parametrize(
    "hint",
    [
        tx.Optional[int],
        tx.Union[int, None],
        tx.Union[str, type],
        tx.Union[int, str],
    ],
)
def test_union_like_is_never_bare_any(hint: tx.Any) -> None:
    assert Converter.get(hint).like() is not tx.Any


def test_none_like() -> None:
    assert Converter.get(type(None)).like() is type(None)  # noqa: E721


def test_type_like() -> None:
    assert Converter.get(type).like() is type


def test_enum_like_reports_what_it_accepts() -> None:
    # locals
    from bagof.converters.enums import ToEnum

    class Colour(enum.Enum):
        RED = 1
        GREEN = 2

    args = tx.get_args(ToEnum(Colour).like())
    # a member, a name, or a value
    assert Colour in args
    assert str in args
    assert int in args


def test_all_any_union_still_reports_any() -> None:
    # If every branch genuinely accepts anything, `Any` is the honest
    # answer -- the filter must not turn that into `Never`.
    assert Converter.get(tx.Union[tx.Any, tx.Any]).like() is tx.Any


# ----------------------------------------------------------------------
# Literal matching
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "hint,value",
    [
        # PEP 586: a literal matches by type as well as by value.
        (tx.Literal[1], True),
        (tx.Literal[1], 1.0),
        (tx.Literal[True], 1),
        (tx.Literal[0], False),
        (tx.Literal["a"], b"a"),
    ],
)
def test_literal_matches_by_type_as_well_as_value(
    hint: tx.Any, value: tx.Any
) -> None:
    with pytest.raises(ConversionError):
        Converter.get(hint)(value)


@pytest.mark.parametrize(
    "hint,value,expected",
    [
        (tx.Literal[1], 1, 1),
        (tx.Literal[True], True, True),
        (tx.Literal[1, "a"], "a", "a"),
        (tx.Literal[None], None, None),
    ],
)
def test_literal_returns_the_matching_literal(
    hint: tx.Any, value: tx.Any, expected: tx.Any
) -> None:
    result = Converter.get(hint)(value)
    assert result == expected
    assert type(result) is type(expected)


# ----------------------------------------------------------------------
# Annotated metadata
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,missing",
    [("ToRegexMatch", "pattern"), ("ToLength", "length")],
)
def test_annotated_converter_class_needing_config_raises(
    cls: str, missing: str
) -> None:
    # locals
    from bagof.converters.collections import ToLength
    from bagof.converters.strings import ToRegexMatch

    # Passed positionally, the annotated type used to be bound as the
    # converter's own first argument (a pattern, a length), producing a
    # converter configured with a type object.
    lookup = {"ToRegexMatch": ToRegexMatch, "ToLength": ToLength}
    with pytest.raises(TypeError, match=missing):
        assert common.ToAnnotated(tx.Annotated[str, lookup[cls]]).converters


def test_annotated_converter_class_taking_hint_first_still_works() -> None:
    # locals
    from bagof.converters.strings import ToString

    converter = common.ToAnnotated(tx.Annotated[str, ToString])
    assert converter(b"hi") == "hi"


def test_regex_converter_can_compose() -> None:
    # locals
    from bagof.converters.strings import ToRegexMatch

    # `ToRegexMatch` took no `compose`, so it always replaced the base
    # converter and could never be chained after it.
    hint = tx.Annotated[str, ToRegexMatch(r"\d+", compose=True)]
    converter = common.ToAnnotated(hint)
    assert len(converter.converters) == 2
    assert converter(b"123") == "123"
    with pytest.raises(ConversionError):
        converter(b"abc")


def test_length_converter_can_compose() -> None:
    # locals
    from bagof.converters.collections import ToLength

    assert ToLength(2, compose=True).compose is True


# ----------------------------------------------------------------------
# Union branch failures
# ----------------------------------------------------------------------


def test_union_surfaces_a_broken_branch() -> None:
    # locals
    from bagof.converters.base import CONVERTERS

    class Broken(Converter[complex, tx.Any]):
        DEFAULT = complex

        def __call__(self, value: tx.Any) -> tx.Any:
            raise TypeError("this branch is broken")

    Converter.register(Broken, complex)
    try:
        # Catching the builtins swallowed genuine bugs in a converter's
        # own code and reported them as "no branch matched".
        with pytest.raises(TypeError, match="this branch is broken"):
            Converter.get(tx.Union[complex, str])(object())
    finally:
        CONVERTERS.pop(complex, None)


def test_union_branch_that_does_not_match_still_falls_through() -> None:
    assert Converter.get(tx.Union[int, str])("abc") == "abc"


def test_union_reports_every_branch_failure_as_a_cause() -> None:
    with pytest.raises(ConversionError) as info:
        Converter.get(tx.Union[int, complex])(object())
    assert len(info.value.causes) == 2


# ----------------------------------------------------------------------
# Union: exact match before coercion
# ----------------------------------------------------------------------


@pytest.mark.parametrize("hint", [tx.Union[int, str], tx.Union[str, int]])
@pytest.mark.parametrize("value", ["5", 5, "abc"])
def test_union_result_does_not_depend_on_branch_order(
    hint: tx.Any, value: tx.Any
) -> None:
    # `Union[int, str]` and `Union[str, int]` denote the same type, but
    # the converter used to give different answers: `"5"` came back as
    # `5` under one spelling and `"5"` under the other.
    result = Converter.get(hint)(value)
    assert result == value
    assert type(result) is type(value)


def test_union_still_coerces_when_nothing_matches() -> None:
    assert Converter.get(tx.Union[int, str])(b"5") == 5
    assert Converter.get(tx.Union[int, float])("5") == 5


def test_optional_short_circuits_none() -> None:
    assert Converter.get(tx.Optional[int])(None) is None
    assert Converter.get(tx.Optional[int])("5") == 5


def test_union_docstring_example() -> None:
    convert = Converter.get(tx.Union[int, str])
    assert convert("5") == "5"
    assert convert(5) == 5
    assert convert(b"5") == 5


# ----------------------------------------------------------------------
# Error wrapping
# ----------------------------------------------------------------------


def test_wrapped_converter_preserves_a_conversion_error() -> None:
    # `TypeConversionError` and `ValueConversionError` inherit the two
    # builtins the wrapper catches, so without an explicit pass-through it
    # downgrades every real failure to a generic "Invalid value."
    with pytest.raises(ConversionError) as info:
        Converter.get(tx.List[str])([1])
    assert "not a string or bytes" in str(info.value)


def test_wrapped_converter_converts_a_bare_builtin() -> None:

    class Broken(Converter[complex, tx.Any]):
        DEFAULT = complex

        def __call__(self, value: tx.Any) -> tx.Any:
            raise ValueError("plain failure")

    wrapped = Converter(int)._wrap_converter(Broken(complex))
    with pytest.raises(ConversionError) as info:
        wrapped(1)
    assert isinstance(info.value.__cause__, ValueError)


def test_register_metadata_is_distinct_from_register() -> None:
    # locals
    from bagof.converters.base import CONVERTERS

    class Marker:
        pass

    @common.ToAnnotated.register_metadata(Marker)
    class MarkedConverter(Converter[int, tx.Any]):
        DEFAULT = int

        def __init__(
            self, marker: tx.Any = None, hint: tx.Any = None
        ) -> None:
            super().__init__(int)
            self.marker = marker

        def __call__(self, value: tx.Any) -> int:
            return 42

    try:
        assert common.ToAnnotated._REGISTRY[Marker] is MarkedConverter
        assert Marker not in CONVERTERS
        assert common.ToAnnotated(tx.Annotated[int, Marker()])("x") == 42
    finally:
        common.ToAnnotated._REGISTRY.pop(Marker, None)


def test_register_alias_still_works() -> None:
    assert (
        common.ToAnnotated.register.__func__
        is common.ToAnnotated.register_metadata.__func__
    )

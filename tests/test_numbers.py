# stdlib
import numbers

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.converters import numbers as conv_numbers
from bagof.converters.base import Converter
from bagof.converters.exceptions import ConversionError

# --- ToNumber ---------------------------------------------------------


@pytest.mark.parametrize(
    "hint,value,expected",
    [
        (int, 1, 1),
        (int, 1.0, 1),
        (int, True, True),
        (float, 1, 1.0),
        (float, 1.5, 1.5),
        (float, "inf", float("inf")),
        (float, "nan", float("nan")),
        (complex, 1, 1+0j),
        (complex, 1.0, 1.0+0j),
        (numbers.Number, 1, 1),
        (numbers.Number, 1.5, 1.5),
    ],
)
def test_number_valid(hint: tx.Any, value: tx.Any, expected: tx.Any) -> None:
    converter = conv_numbers.ToNumber(hint)
    result = converter(value)
    if isinstance(expected, float) and expected != expected:  # NaN check
        assert result != result
    else:
        assert result == expected


@pytest.mark.parametrize(
    "hint,value",
    [
        (int, "abc"),
        (float, [1, 2]),
        (int, None),
    ],
)
def test_number_invalid(hint: tx.Any, value: tx.Any) -> None:
    converter = conv_numbers.ToNumber(hint)
    with pytest.raises((ConversionError, TypeError, ValueError)):
        converter(value)


# --- ToPositive -------------------------------------------------------


@pytest.mark.parametrize("value", [1, 0.1, 1000])
def test_positive_valid(value: tx.Any) -> None:
    converter = conv_numbers.ToPositive()
    assert converter(value) == value


@pytest.mark.parametrize("value", [0, -1, -0.1])
def test_positive_invalid(value: tx.Any) -> None:
    converter = conv_numbers.ToPositive()
    with pytest.raises(ConversionError):
        converter(value)


# --- ToNegative -------------------------------------------------------


@pytest.mark.parametrize("value", [-1, -0.1, -1000])
def test_negative_valid(value: tx.Any) -> None:
    converter = conv_numbers.ToNegative()
    assert converter(value) == value


@pytest.mark.parametrize("value", [0, 1, 0.1])
def test_negative_invalid(value: tx.Any) -> None:
    converter = conv_numbers.ToNegative()
    with pytest.raises(ConversionError):
        converter(value)


# --- ToNonNegative ----------------------------------------------------


@pytest.mark.parametrize("value", [0, 1, 0.1, 1000])
def test_non_negative_valid(value: tx.Any) -> None:
    converter = conv_numbers.ToNonNegative()
    assert converter(value) == value


@pytest.mark.parametrize("value", [-1, -0.1])
def test_non_negative_invalid(value: tx.Any) -> None:
    converter = conv_numbers.ToNonNegative()
    with pytest.raises(ConversionError):
        converter(value)


# --- ToNonPositive ----------------------------------------------------


@pytest.mark.parametrize("value", [0, -1, -0.1])
def test_non_positive_valid(value: tx.Any) -> None:
    converter = conv_numbers.ToNonPositive()
    assert converter(value) == value


@pytest.mark.parametrize("value", [1, 0.1])
def test_non_positive_invalid(value: tx.Any) -> None:
    converter = conv_numbers.ToNonPositive()
    with pytest.raises(ConversionError):
        converter(value)


# --- ToLessThan -------------------------------------------------------


@pytest.mark.parametrize("threshold,value", [(5, 4), (0, -1), (1.0, 0.9)])
def test_less_than_valid(threshold: tx.Any, value: tx.Any) -> None:
    converter = conv_numbers.ToLessThan(threshold)
    assert converter(value) == value


@pytest.mark.parametrize("threshold,value", [(5, 5), (5, 6), (0, 0)])
def test_less_than_invalid(threshold: tx.Any, value: tx.Any) -> None:
    converter = conv_numbers.ToLessThan(threshold)
    with pytest.raises(ConversionError):
        converter(value)


# --- ToLessEqual ------------------------------------------------------


@pytest.mark.parametrize("threshold,value", [(5, 5), (5, 4), (0, -1)])
def test_less_equal_valid(threshold: tx.Any, value: tx.Any) -> None:
    converter = conv_numbers.ToLessEqual(threshold)
    assert converter(value) == value


@pytest.mark.parametrize("threshold,value", [(5, 6), (0, 1)])
def test_less_equal_invalid(threshold: tx.Any, value: tx.Any) -> None:
    converter = conv_numbers.ToLessEqual(threshold)
    with pytest.raises(ConversionError):
        converter(value)


# --- ToGreaterThan ----------------------------------------------------


@pytest.mark.parametrize("threshold,value", [(5, 6), (0, 1), (-1, 0)])
def test_greater_than_valid(threshold: tx.Any, value: tx.Any) -> None:
    converter = conv_numbers.ToGreaterThan(threshold)
    assert converter(value) == value


@pytest.mark.parametrize("threshold,value", [(5, 5), (5, 4)])
def test_greater_than_invalid(threshold: tx.Any, value: tx.Any) -> None:
    converter = conv_numbers.ToGreaterThan(threshold)
    with pytest.raises(ConversionError):
        converter(value)


# --- ToGreaterEqual ---------------------------------------------------


@pytest.mark.parametrize("threshold,value", [(5, 5), (5, 6), (0, 0)])
def test_greater_equal_valid(threshold: tx.Any, value: tx.Any) -> None:
    converter = conv_numbers.ToGreaterEqual(threshold)
    assert converter(value) == value


@pytest.mark.parametrize("threshold,value", [(5, 4), (0, -1)])
def test_greater_equal_invalid(threshold: tx.Any, value: tx.Any) -> None:
    converter = conv_numbers.ToGreaterEqual(threshold)
    with pytest.raises(ConversionError):
        converter(value)


# --- ToInRange --------------------------------------------------------


@pytest.mark.parametrize(
    "mn,mx,inclusive,value",
    [
        (0, 10, True, 0),
        (0, 10, True, 5),
        (0, 10, True, 10),
        (0, 10, False, 1),
        (0, 10, False, 9),
        (0, 10, (True, False), 0),
        (0, 10, (False, True), 10),
    ],
)
def test_in_range_valid(
    mn: tx.Any, mx: tx.Any, inclusive: tx.Any, value: tx.Any
) -> None:
    converter = conv_numbers.ToInRange(mn, mx, inclusive)
    assert converter(value) == value


@pytest.mark.parametrize(
    "mn,mx,inclusive,value",
    [
        (0, 10, True, -1),
        (0, 10, True, 11),
        (0, 10, False, 0),
        (0, 10, False, 10),
        (0, 10, (True, False), 10),
        (0, 10, (False, True), 0),
    ],
)
def test_in_range_invalid(
    mn: tx.Any, mx: tx.Any, inclusive: tx.Any, value: tx.Any
) -> None:
    converter = conv_numbers.ToInRange(mn, mx, inclusive)
    with pytest.raises(ConversionError):
        converter(value)


# --- ToBool -----------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        # strings are parsed by spelling, not truthiness
        ("false", False),
        ("False", False),
        ("FALSE", False),
        (" no ", False),
        ("off", False),
        ("0", False),
        ("f", False),
        ("", False),
        ("true", True),
        ("Yes", True),
        ("on", True),
        ("1", True),
        # non-strings go through numeric coercion
        (0, False),
        (1, True),
        (2, True),
        (0.0, False),
        (True, True),
        (False, False),
    ],
)
def test_bool_valid(value: tx.Any, expected: bool) -> None:
    result = conv_numbers.ToBool()(value)
    assert result is expected


@pytest.mark.parametrize("value", ["maybe", "2", "yesnt", "y e s"])
def test_bool_invalid_strings(value: str) -> None:
    # Unrecognised strings raise rather than defaulting to True.
    with pytest.raises(ConversionError):
        conv_numbers.ToBool()(value)


def test_bool_is_registered() -> None:
    from bagof.converters.base import Converter

    assert Converter.get_class(bool) is conv_numbers.ToBool
    assert Converter.get(bool)("no") is False


def test_bool_like() -> None:
    assert conv_numbers.ToBool().like() == tx.Union[bool, int, str]


# ----------------------------------------------------------------------
# Abstract numeric hints: the fallback ladder
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "hint,value,expected,expected_type",
    [
        # Nothing converts while preserving equality, so the widest type
        # the hint allows is used. `fallbacks[0]` is `bool` -- the
        # *narrowest* -- and turned each of these into `True`.
        (numbers.Integral, "5", 5, int),
        (numbers.Real, "1.5", 1.5, float),
        (numbers.Complex, "1", 1 + 0j, complex),
    ],
)
def test_abstract_hint_falls_back_to_the_widest_type(
    hint: tx.Any, value: tx.Any, expected: tx.Any, expected_type: type
) -> None:
    result = Converter.get(hint)(value)
    assert result == expected
    assert type(result) is expected_type


@pytest.mark.parametrize(
    "hint,value,expected_type",
    [
        # These *do* convert while preserving equality, so the narrowest
        # equality-preserving type wins.
        (numbers.Integral, 5.0, int),
        (numbers.Real, 2, int),
    ],
)
def test_abstract_hint_prefers_an_equality_preserving_type(
    hint: tx.Any, value: tx.Any, expected_type: type
) -> None:
    result = Converter.get(hint)(value)
    assert result == value
    assert type(result) is expected_type


def test_abstract_hint_with_an_unconvertible_value_raises() -> None:
    with pytest.raises(ConversionError):
        Converter.get(numbers.Integral)("not a number")


def test_fallback_ladders_are_ordered_narrowest_first() -> None:
    # The final fallback takes the last entry, so the ordering is
    # load-bearing rather than cosmetic.
    ladders = conv_numbers.ToNumber.FALLBACKS
    assert ladders[numbers.Integral][-1] is int
    assert ladders[numbers.Real][-1] is float
    assert ladders[numbers.Number][-1] is complex


@pytest.mark.parametrize(
    "hint,expected_member",
    [
        (int, numbers.Integral),
        (float, numbers.Real),
        (numbers.Number, numbers.Number),
    ],
)
def test_like_widens_to_the_numeric_tower(
    hint: tx.Any, expected_member: tx.Any
) -> None:
    like = Converter.get(hint).like()
    assert expected_member in tx.get_args(like)


def test_like_is_reentrant_safe() -> None:
    # The re-entrancy guard returns the hint itself on the second visit.
    converter = Converter.get(int)
    assert converter.like((converter.hint,)) is converter.hint


def test_like_without_numpy_returns_the_bare_hint(
    monkeypatch: tx.Any,
) -> None:
    """With no array library installed, `like` is just the numeric hint."""
    monkeypatch.setattr(conv_numbers, "np", None)
    assert Converter.get(numbers.Integral).like() is numbers.Integral
    assert Converter.get(int).like() is int

# stdlib
import numbers

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.converters import numbers as conv_numbers
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

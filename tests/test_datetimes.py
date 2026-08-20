"""Tests for the date and time converters."""

# stdlib
import datetime

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.converters.base import Converter
from bagof.converters.exceptions import ConversionError


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2020-01-01T12:30", datetime.datetime(2020, 1, 1, 12, 30)),
        ("2020-01-01", datetime.datetime(2020, 1, 1)),
        (datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 1)),
    ],
)
def test_datetime_valid(value: tx.Any, expected: tx.Any) -> None:
    assert Converter.get(datetime.datetime)(value) == expected


def test_datetime_from_timestamp() -> None:
    stamp = datetime.datetime(2020, 1, 1, 12).timestamp()
    assert Converter.get(datetime.datetime)(stamp) == datetime.datetime(
        2020, 1, 1, 12
    )


def test_datetime_keeps_naive_naive_and_aware_aware() -> None:
    # No time zone is assumed either way, so nothing is silently shifted.
    naive = Converter.get(datetime.datetime)("2020-01-01T12:00")
    assert naive.tzinfo is None
    aware = Converter.get(datetime.datetime)("2020-01-01T12:00+02:00")
    assert aware.utcoffset() == datetime.timedelta(hours=2)


@pytest.mark.parametrize("value", ["not a date", [1], None, object()])
def test_datetime_invalid(value: tx.Any) -> None:
    with pytest.raises(ConversionError):
        Converter.get(datetime.datetime)(value)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2020-01-01", datetime.date(2020, 1, 1)),
        (datetime.date(2020, 1, 1), datetime.date(2020, 1, 1)),
        (datetime.datetime(2020, 1, 1, 12), datetime.date(2020, 1, 1)),
    ],
)
def test_date_valid(value: tx.Any, expected: tx.Any) -> None:
    assert Converter.get(datetime.date)(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("12:30", datetime.time(12, 30)),
        (datetime.time(12, 30), datetime.time(12, 30)),
        (datetime.datetime(2020, 1, 1, 12, 30), datetime.time(12, 30)),
    ],
)
def test_time_valid(value: tx.Any, expected: tx.Any) -> None:
    assert Converter.get(datetime.time)(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (90, datetime.timedelta(seconds=90)),
        (1.5, datetime.timedelta(seconds=1.5)),
        (datetime.timedelta(days=1), datetime.timedelta(days=1)),
    ],
)
def test_timedelta_valid(value: tx.Any, expected: tx.Any) -> None:
    assert Converter.get(datetime.timedelta)(value) == expected


@pytest.mark.parametrize("value", [True, "90", None])
def test_timedelta_invalid(value: tx.Any) -> None:
    # `bool` is an `int` subclass, but a boolean number of seconds is
    # almost certainly a mistake.
    with pytest.raises(ConversionError):
        Converter.get(datetime.timedelta)(value)


@pytest.mark.parametrize(
    "hint",
    [datetime.datetime, datetime.date, datetime.time, datetime.timedelta],
)
def test_like_is_informative(hint: tx.Any) -> None:
    assert Converter.get(hint).like() is not tx.Any


@pytest.mark.parametrize(
    "hint", [datetime.date, datetime.time]
)
def test_date_and_time_reject_an_unrelated_type(hint: tx.Any) -> None:
    with pytest.raises(ConversionError, match="Cannot convert"):
        Converter.get(hint)(object())

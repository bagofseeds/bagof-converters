"""Converters for date and time types."""

__all__ = ["ToDateTime", "ToDate", "ToTime", "ToTimedelta"]

# stdlib
import datetime

# dependencies
import typing_extensions as tx

# bags
from bagof.core.magic import safe_isinstance
from bagof.hints.typevars.co import T

# locals
from .base import Converter


class ToDateTime(Converter[T, tx.Any], register=datetime.datetime):
    """
    Converter for [`datetime`][datetime.datetime].

    Accepts a `datetime`, an ISO 8601 string, or a POSIX timestamp.

    !!! warning
        A naive string stays naive and an aware string keeps its offset --
        no time zone is assumed either way, so nothing is silently shifted.
        Attach one yourself if your data needs it.

        A numeric timestamp is read as POSIX seconds and interpreted in
        the local time zone, matching
        [`fromtimestamp`][datetime.datetime.fromtimestamp].

    !!! example
        ```pycon
        >>> from bagof.converters import get_converter
        >>> get_converter(datetime.datetime)("2020-01-01T12:30")
        datetime.datetime(2020, 1, 1, 12, 30)
        ```
    """

    DEFAULT = datetime.datetime

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Accept a datetime, an ISO 8601 string, or a timestamp."""
        return tx.Union[datetime.datetime, str, int, float]

    def __call__(self, value: tx.Any) -> T:
        """Convert the value to a datetime."""
        if safe_isinstance(value, datetime.datetime):
            return value
        if safe_isinstance(value, str):
            return self._wrap_converter(
                datetime.datetime.fromisoformat
            )(value)
        if safe_isinstance(value, (int, float)) and not safe_isinstance(
            value, bool
        ):
            return self._wrap_converter(
                datetime.datetime.fromtimestamp
            )(value)
        raise self.type_error(
            value,
            f"Cannot convert a value of type {type(value)} to a datetime.",
        )


class ToDate(Converter[T, tx.Any], register=datetime.date):
    """
    Converter for [`date`][datetime.date].

    Accepts a `date`, a `datetime` (whose date part is taken), or an
    ISO 8601 string.
    """

    DEFAULT = datetime.date

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Accept a date, a datetime, or an ISO 8601 string."""
        return tx.Union[datetime.date, str]

    def __call__(self, value: tx.Any) -> T:
        """Convert the value to a date."""
        if safe_isinstance(value, datetime.datetime):
            return value.date()
        if safe_isinstance(value, datetime.date):
            return value
        if safe_isinstance(value, str):
            return self._wrap_converter(datetime.date.fromisoformat)(value)
        raise self.type_error(
            value,
            f"Cannot convert a value of type {type(value)} to a date.",
        )


class ToTime(Converter[T, tx.Any], register=datetime.time):
    """
    Converter for [`time`][datetime.time].

    Accepts a `time`, a `datetime` (whose time part is taken), or an
    ISO 8601 string.
    """

    DEFAULT = datetime.time

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Accept a time, a datetime, or an ISO 8601 string."""
        return tx.Union[datetime.time, datetime.datetime, str]

    def __call__(self, value: tx.Any) -> T:
        """Convert the value to a time."""
        if safe_isinstance(value, datetime.datetime):
            return value.time()
        if safe_isinstance(value, datetime.time):
            return value
        if safe_isinstance(value, str):
            return self._wrap_converter(datetime.time.fromisoformat)(value)
        raise self.type_error(
            value,
            f"Cannot convert a value of type {type(value)} to a time.",
        )


class ToTimedelta(Converter[T, tx.Any], register=datetime.timedelta):
    """
    Converter for [`timedelta`][datetime.timedelta].

    Accepts a `timedelta` or a number of seconds.
    """

    DEFAULT = datetime.timedelta

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Accept a timedelta or a number of seconds."""
        return tx.Union[datetime.timedelta, int, float]

    def __call__(self, value: tx.Any) -> T:
        """Convert the value to a timedelta."""
        if safe_isinstance(value, datetime.timedelta):
            return value
        if safe_isinstance(value, (int, float)) and not safe_isinstance(
            value, bool
        ):
            return datetime.timedelta(seconds=value)
        raise self.type_error(
            value,
            f"Cannot convert a value of type {type(value)} to a timedelta.",
        )

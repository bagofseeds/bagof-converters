"""Converters for enumeration types."""

__all__ = ["ToEnum"]

# stdlib
import enum

# dependencies
import typing_extensions as tx

# bags
from bagof.hints.typevars.co import T

# locals
from .base import Converter


class ToEnum(Converter[T, tx.Any], register=enum.Enum):
    """
    Converter for [`Enum`][enum.Enum] (and its subclasses).

    A value is resolved first by *value* (``Colour(1)``) and then, failing
    that, by *name* (``Colour["RED"]``). Members are returned unchanged.

    Only [`enum.Enum`][] is registered; [`IntEnum`][enum.IntEnum],
    [`StrEnum`][enum.StrEnum], [`Flag`][enum.Flag] and the rest dispatch to
    it as subclasses (and ``StrEnum`` does not exist before Python 3.11).
    """

    DEFAULT = enum.Enum

    def __call__(self, value: tx.Any) -> T:
        """Convert the value to a member of the enum."""
        origin = self.origin
        if isinstance(value, origin):
            return value  # type: ignore[return-value]
        # By value. ``Flag`` raises ValueError on 3.11+ but TypeError on 3.8
        # (it tries to OR the value), so catch both and fall back to name.
        try:
            return origin(value)  # type: ignore[return-value]
        except (ValueError, TypeError):
            pass
        # By name.
        try:
            return origin[value]  # type: ignore[return-value]
        except (KeyError, TypeError) as e:
            raise self.value_error(
                value, f"Cannot convert {value!r} to {origin.__name__}."
            ) from e

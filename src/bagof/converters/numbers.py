"""Converters for numeric types."""

__all__ = [
    "ToNumber",
    "ToBool",
    "ToPositive",
    "ToNegative",
    "ToNonNegative",
    "ToNonPositive",
    "ToLessThan",
    "ToLessEqual",
    "ToGreaterThan",
    "ToGreaterEqual",
    "ToInRange",
]

# stdlib
import inspect
import numbers

# dependencies
import typing_extensions as tx

# bags
from bagof.core.magic import UNSET, safe_isinstance, safe_issubclass
from bagof.hints.typevars.co import NUMBER

# locals
from ._compat import np
from .base import Converter, _process_reentrant, _trywrap_converter


class ToNumber(Converter[NUMBER, tx.Any], register=numbers.Number):
    """
    Converter for [`numbers.Number`][] types.

    !!! example
        ```pycon
        >>> from bagof.converters import get_converter
        >>> get_converter(int)("42")
        42
        >>> get_converter(float)("1.5")
        1.5
        ```
    """

    DEFAULT = numbers.Number
    FALLBACKS = {
        numbers.Number: (bool, int, float, complex),
        numbers.Real: (bool, int, float),
        numbers.Integral: (bool, int),
    }
    FLOAT_LIKE = ("inf", "infinity", "-inf", "-infinity", "nan")

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Return the input hint for this converter."""
        __reentrant = _process_reentrant(self.hint, __reentrant)
        if not __reentrant:
            return self.hint

        fallback = self.unwrapped
        if np is not None:
            if safe_issubclass(fallback, numbers.Integral):
                return tx.Union[numbers.Integral, np.integer, np.bool_]
            if safe_issubclass(fallback, numbers.Real):
                return tx.Union[numbers.Real, np.floating, np.bool_]
            if safe_issubclass(fallback, numbers.Number):
                return tx.Union[numbers.Number, np.number, np.bool_]
        return fallback

    def __call__(self, value: tx.Any) -> NUMBER:
        """Convert the value to a number."""
        if safe_isinstance(value, str) and value.lower() in self.FLOAT_LIKE:
            value = float(value)
        if safe_isinstance(value, self.hint):
            return value  # type: ignore[return-value]

        # If a concrete type, use it as a converter
        origin = self.origin
        if safe_isinstance(origin, type) and not inspect.isabstract(origin):
            return self._try_convert(value, origin)

        # Otherwise, try the fallbacks
        # -> Only accept output if equality is preserved
        fallbacks = self.FALLBACKS.get(origin, None)
        fallbacks = fallbacks or self.FALLBACKS[numbers.Number]
        for fallback in fallbacks:
            result = self._softtry_convert(value, fallback)
            if result is not None:
                return result  # type: ignore[return-value]

        # Try the most general fallback
        result = self._try_convert(value, fallbacks[0])
        return result  # type: ignore[return-value]

    def _try_convert(self, value: tx.Any, type: type) -> tx.Any:
        return _trywrap_converter(type, self.value_error)(value)

    def _softtry_convert(
        self, value: tx.Any, type: type
    ) -> tx.Optional[tx.Any]:
        try:
            new_value = type(value)
            if new_value == value:
                return new_value
        except (ValueError, TypeError):
            pass
        return None


# --- Bool -------------------------------------------------------------


class ToBool(ToNumber[bool], register=bool):
    """
    Converter for [`bool`][].

    Strings are parsed by spelling rather than by truthiness -- ``"false"``,
    ``"0"``, ``"no"`` and the like become `False`, not `True` (which is what
    plain [`bool`][] does with any non-empty string). Unrecognised strings
    raise, rather than defaulting to `True`.
    """

    DEFAULT = bool
    TRUE_STRINGS = frozenset({"true", "t", "yes", "y", "on", "1"})
    FALSE_STRINGS = frozenset({"false", "f", "no", "n", "off", "0", ""})

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Accept booleans, integers and the recognised string spellings."""
        return tx.Union[bool, int, str]

    def __call__(self, value: tx.Any) -> bool:
        """Convert the value to a bool."""
        if safe_isinstance(value, str):
            key = value.strip().lower()
            if key in self.FALSE_STRINGS:
                return False
            if key in self.TRUE_STRINGS:
                return True
            raise self.value_error(
                value, f"Cannot convert {value!r} to a bool."
            )
        return bool(super().__call__(value))


# --- Sign converters --------------------------------------------------


class ToPositive(ToNumber[NUMBER]):
    """Converter that also checks the value is positive (> 0)."""

    def __call__(self, value: tx.Any) -> NUMBER:
        """Convert the value to a number and check it is positive."""
        value = super().__call__(value)
        if value <= 0:
            raise self.value_error(value, "Expected positive value.")
        return value


class ToNegative(ToNumber[NUMBER]):
    """Converter that also checks the value is negative (< 0)."""

    def __call__(self, value: tx.Any) -> NUMBER:
        """Convert the value to a number and check it is negative."""
        value = super().__call__(value)
        if value >= 0:
            raise self.value_error(value, "Expected negative value.")
        return value


class ToNonNegative(ToNumber[NUMBER]):
    """Converter that also checks the value is non-negative (>= 0)."""

    def __call__(self, value: tx.Any) -> NUMBER:
        """Convert the value to a number and check it is non-negative."""
        value = super().__call__(value)
        if value < 0:
            raise self.value_error(value, "Expected non-negative value.")
        return value


class ToNonPositive(ToNumber[NUMBER]):
    """Converter that also checks the value is non-positive (<= 0)."""

    def __call__(self, value: tx.Any) -> NUMBER:
        """Convert the value to a number and check it is non-positive."""
        value = super().__call__(value)
        if value > 0:
            raise self.value_error(value, "Expected non-positive value.")
        return value


# --- Comparison converters --------------------------------------------


class _ComparatorConverter(ToNumber[NUMBER]):

    def __init__(self, threshold: NUMBER, hint: tx.Any = UNSET) -> None:
        """
        Parameters
        ----------
        threshold : NUMBER
            The threshold value to compare against.
        hint : Any, optional
            The type hint to convert to.
        """
        super().__init__(hint)
        self.threshold = threshold


class ToLessThan(_ComparatorConverter[NUMBER]):
    """Converter that also checks the value is less than a threshold."""

    def __call__(self, value: tx.Any) -> NUMBER:
        """Convert and check value < threshold."""
        value = super().__call__(value)
        if value >= self.threshold:
            raise self.value_error(
                value, f"Expected value less than {self.threshold!r}."
            )
        return value


class ToLessEqual(_ComparatorConverter[NUMBER]):
    """Converter that also checks the value is at most a threshold."""

    def __call__(self, value: tx.Any) -> NUMBER:
        """Convert and check value <= threshold."""
        value = super().__call__(value)
        if value > self.threshold:
            raise self.value_error(
                value,
                f"Expected value less than or equal to {self.threshold!r}.",
            )
        return value


class ToGreaterThan(_ComparatorConverter[NUMBER]):
    """Converter that also checks the value is greater than a threshold."""

    def __call__(self, value: tx.Any) -> NUMBER:
        """Convert and check value > threshold."""
        value = super().__call__(value)
        if value <= self.threshold:
            raise self.value_error(
                value, f"Expected value greater than {self.threshold!r}."
            )
        return value


class ToGreaterEqual(_ComparatorConverter[NUMBER]):
    """Converter that also checks the value is at least a threshold."""

    def __call__(self, value: tx.Any) -> NUMBER:
        """Convert and check value >= threshold."""
        value = super().__call__(value)
        if value < self.threshold:
            raise self.value_error(
                value,
                f"Expected value greater than or equal to {self.threshold!r}.",
            )
        return value


class ToInRange(ToNumber[NUMBER]):
    """Converter that also checks the value is within a range."""

    def __init__(
        self,
        min_value: NUMBER,
        max_value: NUMBER,
        inclusive: tx.Union[bool, tx.Tuple[bool, bool]] = True,
        hint: tx.Any = UNSET,
    ) -> None:
        """
        Parameters
        ----------
        min_value : NUMBER
            The minimum value of the range.
        max_value : NUMBER
            The maximum value of the range.
        inclusive : bool | (bool, bool), optional
            Whether the range is inclusive on both ends.
            If a single boolean is provided, it applies to both ends.
        hint : Any, optional
            The type hint to convert to.
        """
        super().__init__(hint)
        if isinstance(inclusive, bool):
            inclusive = (inclusive, inclusive)
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive

    def __call__(self, value: tx.Any) -> NUMBER:
        """Convert and check the value is in [min_value, max_value]."""
        value = super().__call__(value)
        mn, mx = self.min_value, self.max_value
        lo_inc, hi_inc = self.inclusive
        if lo_inc and hi_inc:
            test = mn <= value <= mx
            lb, ub = "[", "]"
        elif not lo_inc and not hi_inc:
            test = mn < value < mx
            lb, ub = "(", ")"
        elif not lo_inc:
            test = mn < value <= mx
            lb, ub = "(", "]"
        else:
            test = mn <= value < mx
            lb, ub = "[", ")"
        if not test:
            raise self.value_error(
                value, f"Not in range {lb}{mn!r}, {mx!r}{ub}."
            )
        return value

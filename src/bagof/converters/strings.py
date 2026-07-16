"""Converters for string types."""

__all__ = [
    "ToString",
    "ToRegexMatch",
]

# stdlib
import re

# dependencies
import typing_extensions as tx

# bags
from bagof.core.magic import UNSET, safe_isinstance, safe_issubclass
from bagof.hints.typevars.co import STR

# locals
from .base import Converter
from .common import ToAnnotated
from .exceptions import TypeConversionError


class ToString(Converter[STR, tx.Any], register=str):
    """Converter for [`str`][]."""

    DEFAULT = str
    FALLBACK = str

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Accept strings and bytes as input."""
        return tx.Union[str, bytes]

    def __call__(self, value: tx.Any) -> STR:
        """Convert the value to a string."""
        return _to_str(
            value,
            self.hint,
            self.fallback,
            self._wrap_converter,
            self._nostrlike_error(value),
        )

    def _nostrlike_error(self, value: tx.Any) -> TypeConversionError:
        return self.type_error(
            value,
            f"Value of type {type(value)} is not a string or bytes.",
        )


def _to_str(
    value: tx.Any,
    hint: tx.Any,
    fallback: tx.Any,
    wrapper: tx.Callable,
    type_error: TypeConversionError,
) -> tx.Any:
    from bagof.core.magic import get_origin_uw
    input_type = type(value)
    origin = get_origin_uw(hint)

    # Fail for non-string-like values
    if not safe_isinstance(value, str) or safe_isinstance(value, bytes):
        raise type_error

    # Decode bytes
    if safe_isinstance(value, bytes):
        value = value.decode()

    # Select best output type
    if safe_issubclass(input_type, origin):
        output_type = input_type
    else:
        output_type = fallback

    output_type = wrapper(output_type)
    return output_type(value)


@ToAnnotated.register(re.Pattern)
class ToRegexMatch(ToString[STR]):
    """Converter for strings that must match a regex pattern."""

    def __init__(
        self, pattern: tx.Union[str, re.Pattern], hint: tx.Any = UNSET
    ) -> None:
        """
        Parameters
        ----------
        pattern : str | re.Pattern
            The regex pattern to match against.
        hint : Any, optional
            The type hint to convert to.
        """
        super().__init__(hint)
        if not safe_isinstance(pattern, re.Pattern):
            pattern = re.compile(pattern)
        self.pattern = pattern

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.pattern!r})"

    def __call__(self, value: tx.Any) -> STR:
        """Convert the value to a string and check it matches the pattern."""
        value = super().__call__(value)
        if not self.pattern.match(value):
            raise self.value_error(
                value,
                f"Value does not match pattern {self.pattern.pattern!r}.",
            )
        return value

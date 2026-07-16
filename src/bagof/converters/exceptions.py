"""Exceptions raised by converters on conversion error."""

__all__ = ["ConversionError", "ValueConversionError", "TypeConversionError"]

# bags
from bagof.core.magic import MagicError


class ConversionError(MagicError):
    """Base class for all conversion errors."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        if "converter" in kwargs:
            kwargs["this"] = kwargs.pop("converter")
        super().__init__(*args, **kwargs)


class ValueConversionError(ConversionError, ValueError):
    """Raised when conversion fails because of the value of an object."""
    ...


class TypeConversionError(ConversionError, TypeError):
    """Raised when conversion fails because of the type of an object."""
    ...

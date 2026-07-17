__all__: list = []

# dependencies
import typing_extensions as tx

# locals
from .base import Converter

if tx.TYPE_CHECKING:
    from bagof.hints.numpy.typevars.co import DTYPE
    from numpy import dtype, generic
else:
    try:
        from bagof.hints.numpy.typevars.co import DTYPE
        from numpy import dtype, generic
    except ImportError:  # pragma: no cover
        dtype = generic = None  # type: ignore[assignment]

if tx.TYPE_CHECKING or dtype is not None:

    class ToDType(Converter[DTYPE, tx.Any], register=(dtype, generic)):
        """Converter for [`numpy.dtype`][numpy.dtype]."""

        DEFAULT = dtype
        FALLBACK = dtype

        def like(self, __reentrant: tuple = ()) -> tx.Any:
            """Accept any dtype-like value as input."""
            return tx.Any

        def __call__(self, value: tx.Any) -> DTYPE:
            """Convert the value to a numpy dtype."""
            target: tx.Any = None
            from bagof.core.magic import safe_issubclass
            if safe_issubclass(self.origin, generic):
                target = self.origin
            if self.args:
                target = self.args[0]
            try:
                result = dtype(value)
                if target is not None and result.type is not target:
                    raise ValueError(
                        f"dtype {result} is not compatible with {target}"
                    )
                return result  # type: ignore[return-value]
            except (ValueError, TypeError) as e:
                raise self.value_error(
                    value, f"Cannot convert value to dtype {target}"
                ) from e

    __all__ += ["ToDType"]

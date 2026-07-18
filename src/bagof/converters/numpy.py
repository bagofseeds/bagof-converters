__all__: list = []

# dependencies
import typing_extensions as tx

# locals
from ._arrays import ArrayConverter
from .base import Converter

if tx.TYPE_CHECKING:
    # Import the bare module so mkdocstrings resolves the `numpy.*`
    # cross-references in the docstrings below. Type-checking only.
    import numpy  # noqa: F401
    import numpy as np
    from bagof.hints.numpy import ndarray as _hint_ndarray
    from bagof.hints.numpy.typevars.co import DTYPE
    from numpy import dtype, generic
else:
    try:
        import numpy as np
        from bagof.hints.numpy import ndarray as _hint_ndarray
        from bagof.hints.numpy.typevars.co import DTYPE
        from numpy import dtype, generic
    except ImportError:  # pragma: no cover
        np = _hint_ndarray = dtype = generic = None  # type: ignore[assignment]

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

    class ToNDArray(ArrayConverter, register=(np.ndarray, _hint_ndarray)):
        """Converter for [`numpy.ndarray`][]."""

        DEFAULT = np.ndarray
        ARRAY = SCALARS = np
        ARRAY_TYPE = np.ndarray
        HINT_TYPE = _hint_ndarray

    __all__ += ["ToDType", "ToNDArray"]

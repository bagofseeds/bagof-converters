__all__: list = []

# dependencies
import typing_extensions as tx

# bags
from bagof.core.magic import has_module, lazy_import, safe_issubclass

# locals
from ._arrays import ArrayConverter, to_host
from .base import Converter

if tx.TYPE_CHECKING:
    # Import the bare module so mkdocstrings resolves the `numpy.*`
    # cross-references in the docstrings below. Type-checking only.
    import numpy  # noqa: F401
    from bagof.hints.numpy.typevars.co import DTYPE
else:
    # The numpy typevars import numpy; use a plain one at runtime.
    from bagof.hints.typevars.co import T as DTYPE


# numpy is only imported once a numpy converter is actually used: the
# registry keys are forward references, resolved once `numpy` (or the hint
# module) has been imported by the caller -- see `Converter.register`.
if tx.TYPE_CHECKING or has_module("numpy"):

    class ToDType(
        Converter[DTYPE, tx.Any],
        register=(
            tx.ForwardRef("numpy.dtype"),
            tx.ForwardRef("numpy.generic"),
        ),
    ):
        """Converter for [`numpy.dtype`][numpy.dtype]."""

        DEFAULT = lazy_import("numpy.dtype")
        FALLBACK = lazy_import("numpy.dtype")

        def like(self, __reentrant: tuple = ()) -> tx.Any:
            """Accept any dtype-like value as input."""
            return tx.Any

        def __call__(self, value: tx.Any) -> DTYPE:
            """Convert the value to a numpy dtype."""
            import numpy as np

            target: tx.Any = None
            if safe_issubclass(self.origin, np.generic):
                target = self.origin
            if self.args:
                target = self.args[0]
            try:
                result = np.dtype(value)
                if target is not None and result.type is not target:
                    raise ValueError(
                        f"dtype {result} is not compatible with {target}"
                    )
                return result  # type: ignore[return-value]
            except (ValueError, TypeError) as e:
                raise self.value_error(
                    value, f"Cannot convert value to dtype {target}"
                ) from e

    class ToNDArray(
        ArrayConverter,
        register=(
            tx.ForwardRef("numpy.ndarray"),
            tx.ForwardRef("bagof.hints.numpy.ndarray"),
        ),
    ):
        """
        Converter for [`numpy.ndarray`][].

        A lazy (dask) array is computed, and a device (cupy) array is
        copied to the host, so that an array of any library converts.
        """

        DEFAULT = lazy_import("numpy.ndarray")
        ARRAY = SCALARS = lazy_import("numpy")
        ARRAY_TYPE = lazy_import("numpy.ndarray")
        HINT_TYPE = lazy_import("bagof.hints.numpy.ndarray")

        def move(self, value: tx.Any) -> tx.Any:
            """Compute a lazy array, and copy a device array to the host."""
            return to_host(value)

    __all__ += ["ToDType", "ToNDArray"]

"""Converters for cupy array types."""

__all__: list = []

# dependencies
import typing_extensions as tx

# locals
from ._arrays import ArrayConverter

if tx.TYPE_CHECKING:
    # Import the bare module so mkdocstrings resolves the `cupy.*`
    # cross-references in the docstrings below. Type-checking only.
    import cupy  # noqa: F401
    import cupy as cp
    from bagof.hints.cupy import ndarray as _hint_ndarray
else:
    try:
        import cupy as cp
        from bagof.hints.cupy import ndarray as _hint_ndarray
    except ImportError:  # pragma: no cover
        cp = _hint_ndarray = None  # type: ignore[assignment]


if tx.TYPE_CHECKING or cp is not None:  # pragma: no cover
    # cupy needs a CUDA toolchain and cannot be installed on a CPU runner,
    # so this converter is only ever type-checked, never exercised in CI.

    class ToCupyArray(ArrayConverter, register=(cp.ndarray, _hint_ndarray)):
        """Converter for [`cupy.ndarray`][]."""

        DEFAULT = cp.ndarray
        ARRAY = SCALARS = cp
        ARRAY_TYPE = cp.ndarray
        HINT_TYPE = _hint_ndarray

    __all__ += ["ToCupyArray"]

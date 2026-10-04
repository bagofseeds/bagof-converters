"""Converters for dask array types."""

__all__: list = []

# dependencies
import typing_extensions as tx

# locals
from ._arrays import ArrayConverter
from .base import _has_module, _lazy

if tx.TYPE_CHECKING:
    # Import the bare module so mkdocstrings resolves the `dask.array.*`
    # cross-references in the docstrings below. Type-checking only.
    import dask.array  # noqa: F401


# dask is only imported once a dask converter is actually used: the
# registry keys are forward references, resolved once `dask.array` (or the
# hint module) has been imported by the caller -- see `Converter.register`.
if tx.TYPE_CHECKING or _has_module("dask"):

    class ToDaskArray(
        ArrayConverter,
        register=(
            tx.ForwardRef("dask.array.Array"),
            tx.ForwardRef("bagof.hints.dask.Array"),
        ),
    ):
        """Converter for [`dask.array.Array`][]."""

        DEFAULT = _lazy("dask.array.Array")
        ARRAY = _lazy("dask.array")
        # dask arrays carry numpy dtypes, so the scalar tables come from numpy.
        SCALARS = _lazy("numpy")
        ARRAY_TYPE = _lazy("dask.array.Array")
        HINT_TYPE = _lazy("bagof.hints.dask.Array")
        # dask has no ``Array.view(cls)``; a subclass is built via its
        # constructor instead.
        CAN_VIEW = False

    __all__ += ["ToDaskArray"]

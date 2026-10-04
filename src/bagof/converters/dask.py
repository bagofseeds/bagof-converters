"""Converters for dask array types."""

__all__: list = []

# dependencies
import typing_extensions as tx

# bags
from bagof.core.magic import has_module, lazy_import

# locals
from ._arrays import ArrayConverter

if tx.TYPE_CHECKING:
    # Import the bare module so mkdocstrings resolves the `dask.array.*`
    # cross-references in the docstrings below. Type-checking only.
    import dask.array  # noqa: F401


# dask is only imported once a dask converter is actually used: the
# registry keys are forward references, resolved once `dask.array` (or the
# hint module) has been imported by the caller -- see `Converter.register`.
if tx.TYPE_CHECKING or has_module("dask"):

    class ToDaskArray(
        ArrayConverter,
        register=(
            tx.ForwardRef("dask.array.Array"),
            tx.ForwardRef("bagof.hints.dask.Array"),
        ),
    ):
        """Converter for [`dask.array.Array`][]."""

        DEFAULT = lazy_import("dask.array.Array")
        ARRAY = lazy_import("dask.array")
        # dask arrays carry numpy dtypes, so the scalar tables come from numpy.
        SCALARS = lazy_import("numpy")
        ARRAY_TYPE = lazy_import("dask.array.Array")
        HINT_TYPE = lazy_import("bagof.hints.dask.Array")
        # dask has no ``Array.view(cls)``; a subclass is built via its
        # constructor instead.
        CAN_VIEW = False

    __all__ += ["ToDaskArray"]

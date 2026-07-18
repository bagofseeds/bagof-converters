"""Converters for dask array types."""

__all__: list = []

# dependencies
import typing_extensions as tx

# locals
from ._arrays import ArrayConverter

if tx.TYPE_CHECKING:
    # Import the bare module so mkdocstrings resolves the `dask.array.*`
    # cross-references in the docstrings below. Type-checking only.
    import dask.array  # noqa: F401
    import dask.array as da
    import numpy as np
    from bagof.hints.dask import Array as _hint_array
else:
    try:
        import dask.array as da
        import numpy as np
        from bagof.hints.dask import Array as _hint_array
    except ImportError:  # pragma: no cover
        da = np = _hint_array = None  # type: ignore[assignment]


if tx.TYPE_CHECKING or da is not None:

    class ToDaskArray(ArrayConverter, register=(da.Array, _hint_array)):
        """Converter for [`dask.array.Array`][]."""

        DEFAULT = da.Array
        ARRAY = da
        # dask arrays carry numpy dtypes, so the scalar tables come from numpy.
        SCALARS = np
        ARRAY_TYPE = da.Array
        HINT_TYPE = _hint_array
        # dask has no ``Array.view(cls)``; a subclass is built via its
        # constructor instead.
        CAN_VIEW = False

    __all__ += ["ToDaskArray"]

"""Converters for pandas types."""

__all__: list = []

# dependencies
import typing_extensions as tx

# locals
from ._lazy import has_module, lazy_import
from .base import Converter

if tx.TYPE_CHECKING:
    # Import the bare module so mkdocstrings resolves the `pandas.*`
    # cross-references in the docstrings below. Type-checking only.
    import pandas  # noqa: F401


# pandas is only imported once a pandas converter is actually used: the
# registry keys are forward references, resolved once `pandas` has been
# imported by the caller -- see `Converter.register`.
if tx.TYPE_CHECKING or has_module("pandas"):

    class ToDataFrame(
        Converter[tx.Any, tx.Any],
        register=tx.ForwardRef("pandas.DataFrame"),
    ):
        """
        Converter for [`pandas.DataFrame`][].

        !!! note
            An existing frame is returned unchanged (the inherited
            [`__call__`][bagof.converters.base.Converter.__call__]
            passes instances of the target type through rather than
            rebuilding them, so a real frame is never needlessly
            copied); anything else is passed to the
            [`pandas.DataFrame`][] constructor.
        """

        DEFAULT = lazy_import("pandas.DataFrame")

        def like(self, __reentrant: tuple = ()) -> tx.Any:
            """A frame, a mapping of columns, or an iterable of rows."""
            import pandas as pd

            return tx.Union[pd.DataFrame, tx.Mapping, tx.Iterable]

    class ToSeries(
        Converter[tx.Any, tx.Any],
        register=tx.ForwardRef("pandas.Series"),
    ):
        """
        Converter for [`pandas.Series`][].

        Like [`ToDataFrame`][bagof.converters.pandas.ToDataFrame], an
        existing series is passed through unchanged; anything else goes to
        the [`pandas.Series`][] constructor.
        """

        DEFAULT = lazy_import("pandas.Series")

        def like(self, __reentrant: tuple = ()) -> tx.Any:
            """A series, an iterable of values, or a mapping."""
            import pandas as pd

            return tx.Union[pd.Series, tx.Iterable, tx.Mapping]

    __all__ += ["ToDataFrame", "ToSeries"]

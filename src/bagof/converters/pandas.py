"""Converters for pandas types."""

__all__: list = []

# dependencies
import typing_extensions as tx

# locals
from .base import Converter

if tx.TYPE_CHECKING:
    # Import the bare module so mkdocstrings resolves the `pandas.*`
    # cross-references in the docstrings below. Type-checking only.
    import pandas  # noqa: F401
    import pandas as pd
else:
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        pd = None


if tx.TYPE_CHECKING or pd is not None:

    class ToDataFrame(Converter[tx.Any, tx.Any], register=pd.DataFrame):
        """
        Converter for [`pandas.DataFrame`][].

        An existing frame is returned unchanged (the inherited
        [`__call__`][bagof.converters.base.Converter.__call__] passes
        instances of the target type through rather than rebuilding them, so
        a real frame is never needlessly copied); anything else is passed to
        the [`pandas.DataFrame`][] constructor.
        """

        DEFAULT = pd.DataFrame

        def like(self, __reentrant: tuple = ()) -> tx.Any:
            """A frame, a mapping of columns, or an iterable of rows."""
            return tx.Union[pd.DataFrame, tx.Mapping, tx.Iterable]

    class ToSeries(Converter[tx.Any, tx.Any], register=pd.Series):
        """
        Converter for [`pandas.Series`][].

        Like [`ToDataFrame`][bagof.converters.pandas.ToDataFrame], an
        existing series is passed through unchanged; anything else goes to
        the [`pandas.Series`][] constructor.
        """

        DEFAULT = pd.Series

        def like(self, __reentrant: tuple = ()) -> tx.Any:
            """A series, an iterable of values, or a mapping."""
            return tx.Union[pd.Series, tx.Iterable, tx.Mapping]

    __all__ += ["ToDataFrame", "ToSeries"]

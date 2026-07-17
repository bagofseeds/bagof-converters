# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.converters.base import Converter

pd = pytest.importorskip("pandas")

converters_pandas = pytest.importorskip("bagof.converters.pandas")


# --- ToDataFrame ------------------------------------------------------


def test_dataframe_from_mapping_of_columns() -> None:
    result = converters_pandas.ToDataFrame()({"a": [1, 2], "b": [3, 4]})
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["a", "b"]
    assert result.shape == (2, 2)


def test_dataframe_from_iterable_of_rows() -> None:
    result = converters_pandas.ToDataFrame()([{"a": 1}, {"a": 2}])
    assert isinstance(result, pd.DataFrame)
    assert list(result["a"]) == [1, 2]


def test_dataframe_passthrough_preserves_identity() -> None:
    # An existing frame must be returned unchanged -- not rebuilt/copied.
    frame = pd.DataFrame({"x": [1, 2]})
    assert converters_pandas.ToDataFrame()(frame) is frame


def test_dataframe_like() -> None:
    like = converters_pandas.ToDataFrame().like()
    assert like == tx.Union[pd.DataFrame, tx.Mapping, tx.Iterable]


def test_dataframe_is_registered() -> None:
    assert Converter.get_class(pd.DataFrame) is converters_pandas.ToDataFrame


# --- ToSeries ---------------------------------------------------------


def test_series_from_iterable() -> None:
    result = converters_pandas.ToSeries()([1, 2, 3])
    assert isinstance(result, pd.Series)
    assert list(result) == [1, 2, 3]


def test_series_passthrough_preserves_identity() -> None:
    series = pd.Series([1, 2, 3])
    assert converters_pandas.ToSeries()(series) is series


def test_series_like() -> None:
    like = converters_pandas.ToSeries().like()
    assert like == tx.Union[pd.Series, tx.Iterable, tx.Mapping]


def test_series_is_registered() -> None:
    assert Converter.get_class(pd.Series) is converters_pandas.ToSeries


# --- dispatch ---------------------------------------------------------


def test_dispatch_via_get() -> None:
    # The public entry point resolves the right converter for each type.
    assert isinstance(Converter.get(pd.DataFrame)({"a": [1]}), pd.DataFrame)
    assert isinstance(Converter.get(pd.Series)([1, 2]), pd.Series)

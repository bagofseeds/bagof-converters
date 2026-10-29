# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.converters.base import Converter

np = pytest.importorskip("numpy")
da = pytest.importorskip("dask.array")

converters_dask = pytest.importorskip("bagof.converters.dask")
DaskNDArray = pytest.importorskip("bagof.hints.dask").NDArray


def test_dask_from_list() -> None:
    result = converters_dask.ToDaskArray()([1, 2, 3])
    assert isinstance(result, da.Array)
    assert result.compute().tolist() == [1, 2, 3]


def test_dask_passthrough_preserves_identity() -> None:
    arr = da.from_array(np.array([1.0, 2.0]))
    assert converters_dask.ToDaskArray()(arr) is arr


@pytest.mark.parametrize(
    "scalar,value,expected_dtype",
    [
        (np.float64, [1, 2, 3], "float64"),
        (np.int32, [1.9, 2.9], "int32"),
        (float, [1, 2], "float64"),
    ],
)
def test_dask_dtype_coercion(
    scalar: tx.Any, value: tx.Any, expected_dtype: str
) -> None:
    result = Converter.get(DaskNDArray[scalar])(value)
    assert isinstance(result, da.Array)
    assert result.dtype == np.dtype(expected_dtype)
    # the numpy dtype tables are reused for dask arrays
    assert result.compute().dtype == np.dtype(expected_dtype)


def test_dask_from_numpy_array() -> None:
    result = Converter.get(DaskNDArray[np.float64])(np.array([1, 2, 3]))
    assert isinstance(result, da.Array)
    assert result.dtype == np.dtype("float64")


def test_dask_like() -> None:
    from bagof.hints.array import ArrayLike

    assert converters_dask.ToDaskArray().like() is ArrayLike


def test_dask_is_registered() -> None:
    assert Converter.get_class(da.Array) is converters_dask.ToDaskArray


def test_dask_shape_broadcast() -> None:
    hints_dask = pytest.importorskip("bagof.hints.dask")
    hint = hints_dask.Array[
        tx.Tuple[tx.Literal[2], tx.Literal[3]], hints_dask.dtype[np.float64]
    ]
    result = Converter.get(hint)([1, 2, 3])
    assert isinstance(result, da.Array)
    assert result.shape == (2, 3)
    assert result.compute().tolist() == [[1, 2, 3], [1, 2, 3]]


def test_dask_shape_squeeze() -> None:
    hints_dask = pytest.importorskip("bagof.hints.dask")
    hint = hints_dask.Array[tx.Tuple[tx.Literal[3]], tx.Any]
    result = Converter.get(hint)(da.ones((1, 1, 3)))
    assert isinstance(result, da.Array)
    assert result.shape == (3,)


def test_dask_shape_mismatch() -> None:
    from bagof.converters.exceptions import ValueConversionError

    hints_dask = pytest.importorskip("bagof.hints.dask")
    hint = hints_dask.Array[tx.Tuple[tx.Literal[2]], tx.Any]
    with pytest.raises(ValueConversionError):
        Converter.get(hint)([1, 2, 3])


def test_dask_shape_unknown_size_is_accepted() -> None:
    # A dask array may not know its size until it is computed.
    hints_dask = pytest.importorskip("bagof.hints.dask")
    hint = hints_dask.Array[tx.Tuple[tx.Literal[2]], tx.Any]
    arr = da.from_array(np.arange(4), chunks=2)
    arr = arr[arr > 1]
    assert arr.shape[0] != arr.shape[0]  # nan
    assert Converter.get(hint)(arr) is arr

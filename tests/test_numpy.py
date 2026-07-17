# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.converters.base import Converter

np = pytest.importorskip("numpy")

converters_numpy = pytest.importorskip("bagof.converters.numpy")


# --- ToDType ----------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("float32", "float32"),
        ("int64", "int64"),
        (float, "float64"),
        (int, "int64" if np.dtype(int) == np.dtype("int64") else "int32"),
        (bool, "bool"),
    ],
)
def test_dtype_valid(value: tx.Any, expected: str) -> None:
    result = converters_numpy.ToDType()(value)
    assert result == np.dtype(expected)


def test_dtype_accepts_a_scalar_type() -> None:
    assert converters_numpy.ToDType()(np.int64) == np.dtype("int64")


@pytest.mark.parametrize("value", ["not-a-dtype", object()])
def test_dtype_invalid(value: tx.Any) -> None:
    from bagof.converters.exceptions import ConversionError

    with pytest.raises(ConversionError):
        converters_numpy.ToDType()(value)


# ``bagof.hints.numpy.dtype`` is subscriptable on every Python (via a stub
# below 3.9, where ``np.dtype[...]`` itself is not), so it is the portable
# way to spell a parametrised dtype hint.
hints_dtype = pytest.importorskip("bagof.hints.numpy").dtype


def test_dtype_parametrised_match() -> None:
    # ``dtype[float64]`` accepts a value that resolves to float64...
    converter = converters_numpy.ToDType(hints_dtype[np.float64])
    assert converter("float64") == np.dtype("float64")


def test_dtype_parametrised_mismatch() -> None:
    from bagof.converters.exceptions import ConversionError

    # ... and rejects one that does not.
    converter = converters_numpy.ToDType(hints_dtype[np.float64])
    with pytest.raises(ConversionError):
        converter("int32")


def test_dtype_from_scalar_origin() -> None:
    # A concrete scalar-type hint (origin is an ``np.generic`` subclass)
    # constrains the result to exactly that scalar, exercising the
    # ``generic`` branch.
    converter = converters_numpy.ToDType(np.float64)
    assert converter("float64") == np.dtype("float64")
    from bagof.converters.exceptions import ConversionError

    with pytest.raises(ConversionError):
        converter("int32")


def test_dtype_like() -> None:
    assert converters_numpy.ToDType().like() is tx.Any


@pytest.mark.parametrize("hint", [np.dtype, np.generic])
def test_dtype_is_registered(hint: tx.Any) -> None:
    assert Converter.get_class(hint) is converters_numpy.ToDType


# --- ToNDArray --------------------------------------------------------

NDArray = pytest.importorskip("bagof.hints.numpy").NDArray


def test_ndarray_from_list() -> None:
    result = converters_numpy.ToNDArray()([1, 2, 3])
    assert isinstance(result, np.ndarray)
    assert result.tolist() == [1, 2, 3]


def test_ndarray_passthrough_preserves_identity() -> None:
    arr = np.array([1.0, 2.0])
    assert converters_numpy.ToNDArray()(arr) is arr


@pytest.mark.parametrize(
    "scalar,value,expected_dtype",
    [
        (np.float64, [1, 2, 3], "float64"),
        (np.int32, [1.9, 2.9], "int32"),
        (float, [1, 2], "float64"),      # python type -> np.floating
        (int, [1.5, 2.5], None),          # np.integer -> default int
        (np.complex128, [1, 2], "complex128"),
    ],
)
def test_ndarray_dtype_coercion(
    scalar: tx.Any, value: tx.Any, expected_dtype: tx.Any
) -> None:
    result = Converter.get(NDArray[scalar])(value)
    assert isinstance(result, np.ndarray)
    if expected_dtype is not None:
        assert result.dtype == np.dtype(expected_dtype)
    else:
        # python ``int`` maps to np.integer -> the platform default int
        assert np.issubdtype(result.dtype, np.integer)


def test_ndarray_reinterprets_subclass() -> None:
    class MyArray(np.ndarray):
        pass

    result = Converter.get(MyArray)([1, 2, 3])
    assert isinstance(result, MyArray)


def test_ndarray_dtype_already_correct_is_kept() -> None:
    arr = np.array([1, 2, 3], dtype=np.float64)
    result = Converter.get(NDArray[np.float64])(arr)
    assert result.dtype == np.dtype("float64")
    # already an ndarray of the right dtype -> not copied
    assert result is arr


def test_ndarray_like() -> None:
    from bagof.hints.array import ArrayLike

    assert converters_numpy.ToNDArray().like() is ArrayLike


def test_ndarray_is_registered() -> None:
    assert Converter.get_class(np.ndarray) is converters_numpy.ToNDArray

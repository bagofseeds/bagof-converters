# stdlib
import sys

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


def test_ndarray_dtype_coerced_from_another_array() -> None:
    # Already an ndarray, but of the wrong dtype: it is cast, not rebuilt.
    arr = np.array([1.5, 2.5], dtype=np.float64)
    result = Converter.get(NDArray[np.int32])(arr)
    assert result.dtype == np.dtype("int32")
    assert result.tolist() == [1, 2]


def test_ndarray_unparametrised_dtype_leaves_the_dtype_alone() -> None:
    # ``ndarray[Any, dtype]`` names no scalar type, so nothing is coerced.
    hint = pytest.importorskip("bagof.hints.numpy").ndarray[tx.Any, np.dtype]
    result = Converter.get(hint)([1, 2, 3])
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.integer)


# --- ToNDArray: arrays of other libraries -----------------------------


class FakeCupyArray:
    """
    A stand-in for a cupy array: it exposes the CUDA array interface,
    copies itself to the host with ``get()``, and -- like cupy -- refuses
    an implicit conversion to numpy.
    """

    def __init__(self, array: tx.Any) -> None:
        self.array = np.asarray(array)

    @property
    def __cuda_array_interface__(self) -> dict:
        return {}

    def get(self) -> tx.Any:
        return self.array

    def __array__(self, *args, **kwargs) -> tx.Any:
        raise TypeError(
            "Implicit conversion to a NumPy array is not allowed."
        )


class FakeDaskArray:
    """A stand-in for a dask array (the dask collection protocol)."""

    def __init__(self, result: tx.Any) -> None:
        self.result = result

    def __dask_graph__(self) -> dict:
        return {}

    def compute(self) -> tx.Any:
        return self.result


def test_ndarray_from_device_array() -> None:
    result = converters_numpy.ToNDArray()(FakeCupyArray([1, 2, 3]))
    assert isinstance(result, np.ndarray)
    assert result.tolist() == [1, 2, 3]


def test_ndarray_from_device_array_with_dtype() -> None:
    result = Converter.get(NDArray[np.float32])(FakeCupyArray([1, 2, 3]))
    assert result.dtype == np.dtype("float32")


def test_ndarray_from_lazy_array() -> None:
    lazy = FakeDaskArray(np.array([1, 2, 3]))
    assert converters_numpy.ToNDArray()(lazy).tolist() == [1, 2, 3]


def test_ndarray_from_lazy_device_array() -> None:
    # A dask array of cupy chunks computes to a cupy array, which must
    # then be copied to the host.
    lazy = FakeDaskArray(FakeCupyArray([1, 2, 3]))
    assert converters_numpy.ToNDArray()(lazy).tolist() == [1, 2, 3]


def test_ndarray_from_device_array_without_a_copy_method() -> None:
    # A device array that has no known way to the host fails as it would
    # have without the move, as a conversion error.
    from bagof.converters.exceptions import ConversionError

    class Opaque(FakeCupyArray):
        get = None  # type: ignore[assignment]

    with pytest.raises(ConversionError):
        converters_numpy.ToNDArray()(Opaque([1, 2, 3]))


def test_ndarray_from_real_dask_array_of_device_chunks() -> None:
    da = pytest.importorskip("dask.array")
    # One chunk: dask concatenates several with numpy, which the stand-in
    # does not support (real cupy chunks dispatch to cupy).
    lazy = da.from_array(np.arange(4.0), chunks=4).map_blocks(
        FakeCupyArray, meta=np.array([])
    )
    result = converters_numpy.ToNDArray()(lazy)
    assert isinstance(result, np.ndarray)
    assert result.tolist() == [0.0, 1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, 1),
        ([1, 2], [1, 2]),
        (FakeCupyArray([1, 2]), [1, 2]),
    ],
)
def test_to_host(value: tx.Any, expected: tx.Any) -> None:
    from bagof.converters._arrays import to_host

    result = to_host(value)
    assert np.asarray(result).tolist() == expected


def test_compute_leaves_device_arrays_alone() -> None:
    from bagof.converters._arrays import compute

    device = FakeCupyArray([1])
    assert compute(device) is device
    assert compute(FakeDaskArray(device)) is device


# --- ToNDArray: shape hints -------------------------------------------

ndarray = pytest.importorskip("bagof.hints.numpy").ndarray
hints_dtype_ = pytest.importorskip("bagof.hints.numpy").dtype
L = tx.Literal


def _shaped(*shape: tx.Any) -> tx.Any:
    return ndarray[tx.Tuple[shape], hints_dtype_[np.float64]]


@pytest.mark.parametrize(
    "hint,value,shape",
    [
        # matching shapes pass
        (_shaped(L[4], L[4]), np.eye(4), (4, 4)),
        (_shaped(int, int), np.ones((2, 5)), (2, 5)),
        (_shaped(L[1, 3]), [1.0], (1,)),
        # left-pad with singleton dimensions
        (_shaped(int, int), [1.0, 2.0, 3.0], (1, 3)),
        (_shaped(int, int, int), 1.0, (1, 1, 1)),
        # broadcast singletons to a Literal size
        (_shaped(L[4], L[4]), 1.0, (4, 4)),
        (_shaped(L[4], L[4]), [1.0, 2.0, 3.0, 4.0], (4, 4)),
        (_shaped(int, L[3]), np.ones((5, 1)), (5, 3)),
        # strip extra leading singleton dimensions
        (_shaped(L[4], L[4]), np.ones((1, 4, 4)), (4, 4)),
        (_shaped(int), np.ones((1, 1, 3)), (3,)),
        (_shaped(L[2], L[2]), np.ones((1, 1, 1)), (2, 2)),
        (_shaped(), np.ones((1, 1)), ()),
        # variadic shapes
        (_shaped(L[3], ...), np.ones((1, 3)), (3, 3)),
        (
            _shaped(tx.Unpack[tx.Tuple[int, ...]], L[3]),
            np.ones((2, 5, 1)),
            (2, 5, 3),
        ),
        (_shaped(tx.Unpack[tx.Tuple[int, ...]], L[3]), 1.0, (3,)),
    ],
)
def test_ndarray_shape(hint: tx.Any, value: tx.Any, shape: tx.Any) -> None:
    result = Converter.get(hint)(value)
    assert isinstance(result, np.ndarray)
    assert result.shape == shape
    assert result.dtype == np.dtype("float64")


def test_ndarray_broadcast_keeps_the_values() -> None:
    result = Converter.get(_shaped(L[2], L[3]))([1.0, 2.0, 3.0])
    assert result.tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]


def test_ndarray_matching_shape_is_not_copied() -> None:
    arr = np.eye(4)
    assert Converter.get(_shaped(L[4], L[4]))(arr) is arr


@pytest.mark.parametrize(
    "hint,value",
    [
        (_shaped(L[4], L[4]), np.ones(3)),
        (_shaped(L[4], L[4]), np.ones((2, 4, 4))),
        (_shaped(L[4], L[4]), np.ones((1, 2, 4, 4))),
        (_shaped(int), np.ones((2, 2))),
        (_shaped(L[2, 3]), [1.0]),
    ],
)
def test_ndarray_shape_mismatch(hint: tx.Any, value: tx.Any) -> None:
    from bagof.converters.exceptions import ValueConversionError

    with pytest.raises(ValueConversionError, match="shape"):
        Converter.get(hint)(value)


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="numpy types subscript from 3.9"
)
def test_ndarray_shape_and_subclass() -> None:
    class MyArray(np.ndarray):
        pass

    result = Converter.get(MyArray[tx.Tuple[L[2]], tx.Any])(1)
    assert isinstance(result, MyArray)
    assert result.shape == (2,)


def test_ndarray_squeeze_is_a_view() -> None:
    arr = np.ones((1, 4, 4))
    result = Converter.get(_shaped(L[4], L[4]))(arr)
    assert result.shape == (4, 4)
    assert np.shares_memory(result, arr)


def test_ndarray_shape_from_device_array() -> None:
    result = Converter.get(_shaped(L[2], L[2]))(FakeCupyArray([1.0, 2.0]))
    assert result.tolist() == [[1.0, 2.0], [1.0, 2.0]]


@pytest.mark.parametrize(
    "hint", [NDArray, ndarray[tx.Any, hints_dtype_[tx.Any]]]
)
def test_ndarray_unbound_dtype_leaves_the_dtype_alone(hint: tx.Any) -> None:
    # A `TypeVar` (the bare `NDArray` alias) or `Any` names no scalar type.
    result = Converter.get(hint)([1, 2])
    assert result.tolist() == [1, 2]
    assert np.issubdtype(result.dtype, np.integer)

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

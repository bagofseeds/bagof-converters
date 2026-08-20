"""Tests for the standalone scalar converters."""

# stdlib
import pathlib
import uuid

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.converters.base import Converter
from bagof.converters.exceptions import ConversionError

UUID_STR = "12345678-1234-5678-1234-567812345678"


@pytest.mark.parametrize(
    "value",
    [
        UUID_STR,
        UUID_STR.replace("-", ""),
        uuid.UUID(UUID_STR),
        uuid.UUID(UUID_STR).bytes,
        uuid.UUID(UUID_STR).int,
    ],
)
def test_uuid_valid(value: tx.Any) -> None:
    assert Converter.get(uuid.UUID)(value) == uuid.UUID(UUID_STR)


@pytest.mark.parametrize("value", ["not a uuid", [1], None, 1.5])
def test_uuid_invalid(value: tx.Any) -> None:
    with pytest.raises(ConversionError):
        Converter.get(uuid.UUID)(value)


@pytest.mark.parametrize("value", ["/tmp/x", pathlib.Path("/tmp/x")])
def test_path_valid(value: tx.Any) -> None:
    assert Converter.get(pathlib.Path)(value) == pathlib.Path("/tmp/x")


def test_path_result_matches_the_hinted_class() -> None:
    result = Converter.get(pathlib.PurePosixPath)("/tmp/x")
    assert type(result) is pathlib.PurePosixPath


def test_path_accepts_os_pathlike() -> None:

    class Fake:
        def __fspath__(self) -> str:
            return "/tmp/x"

    assert Converter.get(pathlib.Path)(Fake()) == pathlib.Path("/tmp/x")


@pytest.mark.parametrize("value", [5, None, [1]])
def test_path_invalid(value: tx.Any) -> None:
    with pytest.raises(ConversionError):
        Converter.get(pathlib.Path)(value)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("abc", b"abc"),
        (b"abc", b"abc"),
        (bytearray(b"abc"), b"abc"),
        (memoryview(b"abc"), b"abc"),
        ("é", "é".encode()),
    ],
)
def test_bytes_valid(value: tx.Any, expected: bytes) -> None:
    result = Converter.get(bytes)(value)
    assert result == expected
    assert type(result) is bytes


@pytest.mark.parametrize("value", [5, None, [1]])
def test_bytes_invalid(value: tx.Any) -> None:
    with pytest.raises(ConversionError):
        Converter.get(bytes)(value)


@pytest.mark.parametrize("hint", [uuid.UUID, pathlib.Path, bytes])
def test_like_is_informative(hint: tx.Any) -> None:
    assert Converter.get(hint).like() is not tx.Any


# ----------------------------------------------------------------------
# ToSlice
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (slice(1, 5), slice(1, 5)),
        (None, slice(None)),
        (5, slice(None, 5, None)),           # `slice(n)` is `[:n]`
        ([3], slice(None, 3, None)),
        (["1", "5"], slice(1, 5)),           # components are converted
        (("1", "5", "2"), slice(1, 5, 2)),
        ([None, 5], slice(None, 5, None)),
        ([None, None, -1], slice(None, None, -1)),
    ],
)
def test_slice_valid(value: tx.Any, expected: slice) -> None:
    result = Converter.get(slice)(value)
    assert result == expected
    assert type(result) is slice


@pytest.mark.parametrize(
    "value",
    [
        "x",             # any string used to become the `stop`
        "1:5",           # the textual form is not supported
        {"a": 1},
        [1, 2, 3, 4],    # too many components
        [],              # too few
        1.5,
        True,            # a bool is an int, but not a sensible bound
        object(),
    ],
)
def test_slice_invalid(value: tx.Any) -> None:
    # Without a dedicated converter these all fell through to the base
    # one, which called `slice(value)` and "succeeded" with nonsense --
    # `(1, 2)` became `slice(None, (1, 2), None)`.
    with pytest.raises(ConversionError):
        Converter.get(slice)(value)


def test_slice_component_that_is_not_an_integer_raises() -> None:
    with pytest.raises(ConversionError):
        Converter.get(slice)(["a", "b"])


def test_slice_like_is_informative() -> None:
    assert Converter.get(slice).like() is not tx.Any


def test_slice_is_registered() -> None:
    # locals
    from bagof.converters.misc import ToSlice

    assert isinstance(Converter.get(slice), ToSlice)

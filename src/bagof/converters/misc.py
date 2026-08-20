"""Converters for standalone scalar types (UUID, Path, ...)."""

__all__ = ["ToUUID", "ToPath", "ToBytes"]

# stdlib
import os
import pathlib
import uuid

# dependencies
import typing_extensions as tx

# bags
from bagof.core.magic import safe_isinstance
from bagof.hints.typevars.co import T

# locals
from .base import Converter


class ToUUID(Converter[T, tx.Any], register=uuid.UUID):
    """
    Converter for [`UUID`][uuid.UUID].

    Accepts a `UUID`, a string in any form [`UUID`][uuid.UUID] accepts,
    16 raw bytes, or an integer.

    !!! example
        ```pycon
        >>> from bagof.converters import get_converter
        >>> get_converter(uuid.UUID)("12345678-1234-5678-1234-567812345678")
        UUID('12345678-1234-5678-1234-567812345678')
        ```
    """

    DEFAULT = uuid.UUID

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Accept a UUID, a string, 16 bytes, or an integer."""
        return tx.Union[uuid.UUID, str, bytes, int]

    def __call__(self, value: tx.Any) -> T:
        """Convert the value to a UUID."""
        if safe_isinstance(value, uuid.UUID):
            return value
        if safe_isinstance(value, str):
            return self._wrap_converter(uuid.UUID)(value)
        if safe_isinstance(value, bytes):
            return self._wrap_converter(
                lambda v: uuid.UUID(bytes=v)
            )(value)
        if safe_isinstance(value, int) and not safe_isinstance(value, bool):
            return self._wrap_converter(lambda v: uuid.UUID(int=v))(value)
        raise self.type_error(
            value,
            f"Cannot convert a value of type {type(value)} to a UUID.",
        )


class ToPath(Converter[T, tx.Any], register=pathlib.PurePath):
    """
    Converter for [`Path`][pathlib.Path] and the other
    [`pathlib`][] classes.

    Accepts a path, a string, or anything implementing
    [`__fspath__`][os.PathLike].

    !!! note
        The result is an instance of the hinted class, so a
        [`PurePosixPath`][pathlib.PurePosixPath] hint yields one of those
        rather than the platform default.
    """

    DEFAULT = pathlib.Path

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Accept a path, a string, or an `os.PathLike`."""
        return tx.Union[pathlib.PurePath, str, os.PathLike]

    def __call__(self, value: tx.Any) -> T:
        """Convert the value to a path of the hinted class."""
        origin = self.origin
        if safe_isinstance(value, origin):
            return value
        if safe_isinstance(value, (str, pathlib.PurePath)) or hasattr(
            value, "__fspath__"
        ):
            return self._wrap_converter(origin)(value)
        raise self.type_error(
            value,
            f"Cannot convert a value of type {type(value)} to a path.",
        )


class ToBytes(Converter[T, tx.Any], register=bytes):
    """
    Converter for [`bytes`][].

    The counterpart to [`ToString`][bagof.converters.strings.ToString],
    which accepts bytes but only ever produces a string. A `str` is
    encoded as UTF-8; a buffer (`bytearray`, `memoryview`) is copied.
    """

    DEFAULT = bytes
    FALLBACK = bytes
    ENCODING = "utf-8"

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Accept bytes, a buffer, or a string."""
        return tx.Union[bytes, bytearray, memoryview, str]

    def __call__(self, value: tx.Any) -> T:
        """Convert the value to bytes."""
        if safe_isinstance(value, str):
            return self._wrap_converter(
                lambda v: v.encode(self.ENCODING)
            )(value)
        if safe_isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)
        raise self.type_error(
            value,
            f"Value of type {type(value)} is not bytes or a string.",
        )

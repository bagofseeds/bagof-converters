"""Converters for collection types (list, tuple, dict, etc.)"""

__all__ = [
    "ToIterable",
    "ToSequence",
    "ToMapping",
    "ToTuple",
    "ToLength",
]

# stdlib
from functools import partial
import inspect
from collections import abc

# dependencies
import typing_extensions as tx

# bags
from bagof.core.magic import (
    UNSET,
    get_args_uw,
    get_origin_uw,
    issubscriptable,
    safe_issubclass,
)
from bagof.hints.typevars.co import ITERABLE, MAPPING, SEQUENCE, TUPLE

# locals
from .base import Converter, _process_reentrant
from .exceptions import ValueConversionError


def _type_to_hint(x: tx.Any) -> tx.Any:
    """Convert a concrete type to a subscriptable type hint if needed."""
    if issubscriptable(x):
        return x
    name = getattr(x, "__name__", "").split(".")[-1].capitalize()
    if hasattr(tx, name):
        return getattr(tx, name)
    return x


# --- Iterable ---------------------------------------------------------


class ToIterable(
    Converter[ITERABLE, tx.Any], register=abc.Iterable
):
    """Converter for [`abc.Iterable`][]."""

    DEFAULT = abc.Iterable
    FALLBACK = abc.Iterable

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Return the input hint for this converter."""
        return _like_iterable(self.hint, __reentrant)

    def __call__(self, value: tx.Any) -> ITERABLE:
        """Convert the value to an iterable, converting each element."""
        return _to_iterable(
            value,
            self.hint,
            self.fallback,
            self._wrap_converter,
        )


def _like_iterable(hint: tx.Any, __reentrant: tuple = ()) -> tx.Any:
    __reentrant = _process_reentrant(hint, __reentrant)
    if not __reentrant:
        return hint

    origin = get_origin_uw(hint)
    args = get_args_uw(hint)
    if ... in args:
        args = args[:1]
    if safe_issubclass(origin, abc.Mapping):
        args = (tx.Tuple[args],)
    args = tuple(
        Converter.get(arg).like(__reentrant)
        for arg in args
    )
    args = tuple(arg for arg in args if arg is not UNSET)
    return tx.Iterable[args] if args else tx.Iterable


def _to_iterable(
    value: tx.Any, hint: tx.Any, fallback: tx.Any, wrapper: tx.Callable
) -> tx.Any:
    input_type = type(value)
    origin = get_origin_uw(hint)
    args = get_args_uw(hint)

    if args:
        arg_converter = wrapper(Converter.get(args[0]))
        value = map(arg_converter, value)

    if safe_issubclass(input_type, origin):
        output_type = input_type
    else:
        output_type = fallback

    if not inspect.isabstract(output_type):
        output_type = wrapper(output_type)
        return output_type(value)

    return value


# --- Sequence ---------------------------------------------------------


class ToSequence(
    Converter[SEQUENCE, tx.Any], register=abc.Sequence
):
    """Converter for [`abc.Sequence`][]."""

    DEFAULT = abc.Sequence
    FALLBACK = list

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Return the input hint for this converter."""
        return _like_sequence(self.hint, __reentrant)

    def __call__(self, value: tx.Any) -> SEQUENCE:
        """Convert the value to a sequence, converting each element."""
        return _to_sequence(
            value,
            self.hint,
            self.fallback,
            self._wrap_converter,
        )


def _like_sequence(hint: tx.Any, __reentrant: tuple = ()) -> tx.Any:
    __reentrant = _process_reentrant(hint, __reentrant)
    if not __reentrant:
        return hint

    args = get_args_uw(hint)
    args = tuple(
        Converter.get(arg).like(__reentrant)
        for arg in args
    )
    args = tuple(arg for arg in args if arg is not UNSET)
    return tx.Iterable[args] if args else tx.Iterable


def _to_sequence(
    value: tx.Any, hint: tx.Any, fallback: tx.Any, wrapper: tx.Callable
) -> tx.Any:
    input_type = type(value)
    origin = get_origin_uw(hint)
    args = get_args_uw(hint)

    if args:
        converter = wrapper(Converter.get(args[0]))
        mapped_converter = wrapper(partial(map, converter))
        value = mapped_converter(value)

    if safe_issubclass(input_type, origin):
        output_type = input_type
    else:
        output_type = fallback
    output_type = wrapper(output_type)
    return output_type(value)


# --- Mapping ----------------------------------------------------------


class ToMapping(
    Converter[MAPPING, tx.Any], register=abc.Mapping
):
    """Converter for [`abc.Mapping`][]."""

    DEFAULT = abc.Mapping
    FALLBACK = dict

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Return the input hint for this converter."""
        return _like_mapping(self.hint, __reentrant)

    def __call__(self, value: tx.Any) -> MAPPING:
        """Convert the value to a mapping, converting each key and value."""
        return _to_mapping(
            value,
            self.hint,
            self.fallback,
            self._wrap_converter,
        )


def _like_mapping(hint: tx.Any, __reentrant: tuple = ()) -> tx.Any:
    __reentrant = _process_reentrant(hint, __reentrant)
    if not __reentrant:
        return hint

    args = get_args_uw(hint)
    if args:
        args = tuple(
            Converter.get(arg).like(__reentrant)
            for arg in args
        )
        args = tuple(arg for arg in args if arg is not UNSET)
        return tx.Union[
            tx.Iterable[tx.Tuple[args]],
            tx.Mapping[args]
        ]

    return tx.Union[
        tx.Iterable[tx.Tuple[tx.Any, tx.Any]],
        tx.Mapping[tx.Any, tx.Any],
    ]


def _to_mapping(
    value: tx.Any, hint: tx.Any, fallback: tx.Any, wrapper: tx.Callable
) -> tx.Any:
    input_type = type(value)
    origin = get_origin_uw(hint)
    args = get_args_uw(hint)

    if args:
        key_converter = wrapper(Converter.get(args[0]))
        val_converter = wrapper(Converter.get(args[1]))
        if isinstance(value, abc.Mapping):
            value = value.items()
        value = {key_converter(k): val_converter(v) for k, v in value}

    if safe_issubclass(input_type, origin):
        output_type = input_type
    else:
        output_type = fallback
    output_type = wrapper(output_type)
    return output_type(value)


# --- Tuple ------------------------------------------------------------


class ToTuple(Converter[TUPLE, tx.Any], register=tuple):
    """Converter for [`tuple`][]."""

    DEFAULT = tuple

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Return the input hint for this converter."""
        return _like_tuple(self.hint, __reentrant)

    def __call__(self, value: tx.Any) -> TUPLE:
        """Convert the value to a tuple, converting each element."""
        args = get_args_uw(self.hint)
        return _to_tuple(
            value,
            self.hint,
            self.fallback,
            self._wrap_converter,
            self._length_error(value, len(args)),
        )

    def _length_error(
        self, value: tx.Any, target: int
    ) -> ValueConversionError:
        message = (
            f"Expected iterable of length {target}, "
            f"got {len(value) if hasattr(value, '__len__') else '?'}."
        )
        return self.value_error(value, message)


def _like_tuple(hint: tx.Any, __reentrant: tuple = ()) -> tx.Any:
    __reentrant = _process_reentrant(hint, __reentrant)
    if not __reentrant:
        return hint

    origin = get_origin_uw(hint)
    args = get_args_uw(hint)
    if ... in args:
        origin = _type_to_hint(origin)
        if issubscriptable(origin):
            origin = origin[args[0]]
        from .collections import ToIterable
        return ToIterable(origin).like(__reentrant)
    args = tuple(
        Converter.get(arg).like(__reentrant)
        for arg in args
    )
    args = tuple(tx.Any if arg is UNSET else arg for arg in args)
    return tx.Tuple[args] if args else tx.Tuple


def _to_tuple(
    value: tx.Any, hint: tx.Any, fallback: tx.Any,
    wrapper: tx.Callable, length_error: tx.Any,
) -> tx.Any:
    input_type = type(value)
    origin = get_origin_uw(hint)
    args = get_args_uw(hint)

    if args:
        if len(args) == 2 and args[1] is Ellipsis:
            converter = wrapper(Converter.get(args[0]))
            value = map(converter, value)
        else:
            value = tuple(value)
            if len(value) != len(args):
                raise length_error
            converters = map(wrapper, map(Converter.get, args))
            value = (
                converter(val)
                for val, converter in zip(value, converters)
            )

    if safe_issubclass(input_type, origin):
        output_type = input_type
    else:
        output_type = fallback
    output_type = wrapper(output_type)
    return output_type(value)


# --- Length ------------------------------------------------------------


class ToLength(ToSequence[ITERABLE]):
    """Converter for sequences of a fixed length."""

    def __init__(
        self,
        length: int,
        hint: tx.Any = UNSET,
    ) -> None:
        """
        Parameters
        ----------
        length : int
            The expected length of the sequence.
        hint : Any, optional
            The type hint to convert to.
        """
        super().__init__(hint)
        self.length = length

    def __call__(self, value: tx.Any) -> ITERABLE:
        """Convert the value to a sequence and check its length."""
        value = super().__call__(value)
        value = value[:self.length]
        if len(value) != self.length:
            raise self.value_error(
                value,
                f"Expected sequence of length {self.length}, "
                f"got {len(value)}.",
            )
        return value  # type: ignore[return-value]

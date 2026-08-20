"""Converters for collection types (list, tuple, dict, etc.)."""

__all__ = [
    "ToIterable",
    "ToSequence",
    "ToSet",
    "ToMutableSet",
    "ToMapping",
    "ToTuple",
    "ToNamedTuple",
    "ToLength",
]

# stdlib
import collections
import inspect
from collections import abc
from functools import partial

# dependencies
import typing_extensions as tx

# bags
from bagof.core.magic import (
    UNSET,
    get_args_uw,
    get_origin_uw,
    issubscriptable,
    safe_isinstance,
    safe_issubclass,
)
from bagof.hints.typevars.co import ITERABLE, MAPPING, SEQUENCE, TUPLE

# locals
from .base import Converter, _process_reentrant
from .exceptions import ValueConversionError


def _mapping_args(
    args: tx.Tuple[tx.Any, ...], origin: tx.Any
) -> tx.Tuple[tx.Any, tx.Any]:
    """
    The (key, value) hints of a mapping, whatever arity it was written at.

    Most mappings carry both (`Dict[str, int]`), but some fix their value
    type and take only a key: `Counter[K]` is a `Mapping[K, int]`, and
    indexing `args[1]` on one used to raise a bare `IndexError`.
    """
    if len(args) >= 2:
        return args[0], args[1]
    if not args:
        return tx.Any, tx.Any
    return args[0], _IMPLIED_VALUE_HINT.get(origin, tx.Any)


_IMPLIED_VALUE_HINT: tx.Dict[tx.Any, tx.Any] = {collections.Counter: int}


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
        return _like_iterable(self.unwrapped, __reentrant)

    def __call__(self, value: tx.Any) -> ITERABLE:
        """Convert the value to an iterable, converting each element."""
        return _to_iterable(
            value,
            self.unwrapped,
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
    origin = get_origin_uw(hint)
    args = get_args_uw(hint)

    # Already a valid instance and no element conversion to do: pass it
    # through unchanged. Rebuilding would needlessly copy a container, and
    # would fail outright on a single-pass iterator (a generator cannot be
    # reconstructed from itself).
    if not args and safe_isinstance(value, origin):
        return value

    input_type = type(value)
    # A single-pass iterator (generator, map, zip, ...) yields itself from
    # ``__iter__`` and so cannot be rebuilt from its own contents; it can
    # never be the output container type.
    one_shot = safe_isinstance(value, abc.Iterator)

    if args:
        arg_converter = wrapper(Converter.get(args[0]))
        value = map(arg_converter, value)

    if (
        not one_shot
        and not inspect.isabstract(input_type)
        and safe_issubclass(input_type, origin)
    ):
        output_type = input_type
    else:
        output_type = fallback

    if inspect.isabstract(output_type):
        # An abstract fallback (e.g. ``abc.Iterable``) has no constructor, so
        # return the value as-is -- lazily, when elements are being mapped.
        return value

    return wrapper(output_type)(value)


# --- Sequence ---------------------------------------------------------


class ToSequence(
    Converter[SEQUENCE, tx.Any], register=abc.Sequence
):
    """Converter for [`abc.Sequence`][]."""

    DEFAULT = abc.Sequence
    FALLBACK = list

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Return the input hint for this converter."""
        return _like_sequence(self.unwrapped, __reentrant)

    def __call__(self, value: tx.Any) -> SEQUENCE:
        """Convert the value to a sequence, converting each element."""
        return _to_sequence(
            value,
            self.unwrapped,
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
    origin = get_origin_uw(hint)
    args = get_args_uw(hint)

    # An already-valid sequence with nothing to convert passes through
    # unchanged rather than being copied.
    if not args and safe_isinstance(value, origin):
        return value

    input_type = type(value)

    if args:
        converter = wrapper(Converter.get(args[0]))
        mapped_converter = wrapper(partial(map, converter))
        value = mapped_converter(value)

    if not inspect.isabstract(input_type) and safe_issubclass(
        input_type, origin
    ):
        output_type = input_type
    else:
        output_type = fallback
    output_type = wrapper(output_type)
    return output_type(value)


# --- Set --------------------------------------------------------------


class ToSet(ToIterable, register=abc.Set):
    """
    Converter for [`abc.Set`][collections.abc.Set].

    Sets convert element-wise like any other iterable.

    !!! note
        The only difference from
        [`ToIterable`][bagof.converters.collections.ToIterable] is the
        concrete fallback, so that an abstract ``Set[int]`` hint still
        produces a real container ([`frozenset`][], since ``abc.Set`` is
        immutable) rather than a bare iterator.
    """

    DEFAULT = abc.Set
    FALLBACK = frozenset


class ToMutableSet(ToSet, register=abc.MutableSet):
    """Converter for [`abc.MutableSet`][collections.abc.MutableSet]."""

    DEFAULT = abc.MutableSet
    FALLBACK = set


# --- Mapping ----------------------------------------------------------


class ToMapping(
    Converter[MAPPING, tx.Any], register=abc.Mapping
):
    """Converter for [`abc.Mapping`][]."""

    DEFAULT = abc.Mapping
    FALLBACK = dict

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Return the input hint for this converter."""
        return _like_mapping(self.unwrapped, __reentrant)

    def __call__(self, value: tx.Any) -> MAPPING:
        """Convert the value to a mapping, converting each key and value."""
        return _to_mapping(
            value,
            self.unwrapped,
            self.fallback,
            self._wrap_converter,
        )


def _like_mapping(hint: tx.Any, __reentrant: tuple = ()) -> tx.Any:
    __reentrant = _process_reentrant(hint, __reentrant)
    if not __reentrant:
        return hint

    args = get_args_uw(hint)
    if args:
        args = _mapping_args(args, get_origin_uw(hint))
        args = tuple(
            Converter.get(arg).like(__reentrant)
            for arg in args
        )
        args = tuple(arg for arg in args if arg is not UNSET)
        if len(args) == 2:
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
    origin = get_origin_uw(hint)
    args = get_args_uw(hint)

    # An already-valid mapping with nothing to convert passes through
    # unchanged rather than being copied.
    if not args and safe_isinstance(value, origin):
        return value

    input_type = type(value)

    if args:
        key_hint, val_hint = _mapping_args(args, origin)
        key_converter = wrapper(Converter.get(key_hint))
        val_converter = wrapper(Converter.get(val_hint))
        if isinstance(value, abc.Mapping):
            value = value.items()
        value = {key_converter(k): val_converter(v) for k, v in value}

    if not inspect.isabstract(input_type) and safe_issubclass(
        input_type, origin
    ):
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
        if _is_namedtuple(get_origin_uw(self.unwrapped)):
            return ToNamedTuple(self.hint).like(__reentrant)
        return _like_tuple(self.unwrapped, __reentrant)

    def __call__(self, value: tx.Any) -> TUPLE:
        """Convert the value to a tuple, converting each element."""
        if _is_namedtuple(get_origin_uw(self.unwrapped)):
            # A NamedTuple is a `tuple` subclass with no `__args__`, so it
            # would otherwise look zero-length and reject everything.
            return ToNamedTuple(self.hint)(value)
        args = self.args
        return _to_tuple(
            value,
            self.unwrapped,
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


def _is_namedtuple(origin: tx.Any) -> bool:
    """Whether `origin` is a NamedTuple class rather than a plain tuple."""
    return (
        isinstance(origin, type)
        and issubclass(origin, tuple)
        and hasattr(origin, "_fields")
        and hasattr(origin, "_field_defaults")
    )


class ToNamedTuple(Converter[TUPLE, tx.Any]):
    """
    Converter for [`NamedTuple`][typing.NamedTuple] subclasses.

    Accepts a mapping keyed by field name, or any sequence in field
    order, and converts each field through its declared type.

    !!! note
        A NamedTuple is a plain `tuple` subclass at runtime, so it cannot
        be a registry key of its own;
        [`ToTuple`][bagof.converters.collections.ToTuple] recognises one
        and delegates here.

    !!! example
        ```pycon
        >>> import typing_extensions as tx
        >>> from bagof.converters import get_converter
        >>> class Point(tx.NamedTuple):
        ...     x: int
        ...     y: int
        >>> get_converter(Point)({"x": "1", "y": "2"})
        Point(x=1, y=2)
        >>> get_converter(Point)(["1", "2"])
        Point(x=1, y=2)
        ```
    """

    DEFAULT = tuple

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Accept an instance, a mapping by field name, or a sequence."""
        origin = get_origin_uw(self.unwrapped)
        return tx.Union[
            origin,
            tx.Mapping[str, tx.Any],
            tx.Sequence[tx.Any],
        ]

    def __call__(self, value: tx.Any) -> TUPLE:
        """Convert the value field by field."""
        cls = get_origin_uw(self.unwrapped)
        fields = cls._fields
        defaults = cls._field_defaults
        hints = tx.get_type_hints(cls, include_extras=True)

        if safe_isinstance(value, abc.Mapping):
            given = dict(value)
            unknown = set(given) - set(fields)
            if unknown:
                raise self.value_error(
                    value,
                    f"Unexpected field(s) {sorted(unknown)!r} "
                    f"for {cls.__name__}.",
                )
        elif safe_isinstance(value, abc.Iterable) and not safe_isinstance(
            value, (str, bytes)
        ):
            items = list(value)
            if len(items) > len(fields):
                raise self.value_error(
                    value,
                    f"Expected at most {len(fields)} values for "
                    f"{cls.__name__}, got {len(items)}.",
                )
            given = dict(zip(fields, items))
        else:
            raise self.type_error(
                value,
                f"Cannot build {cls.__name__} from "
                f"a value of type {type(value)}.",
            )

        values = {}
        for name in fields:
            if name in given:
                converter = Converter.get(hints.get(name, tx.Any))
                values[name] = self._wrap_converter(converter)(given[name])
            elif name in defaults:
                values[name] = defaults[name]
            else:
                raise self.value_error(
                    value, f"Missing field {name!r} for {cls.__name__}."
                )
        return cls(**values)


# --- Length ------------------------------------------------------------


class ToLength(ToSequence[ITERABLE]):
    """
    Converter for sequences of a fixed length.

    A longer input is **truncated** to the requested length, and a
    shorter one is refused. Coercing to a fixed length is a conversion
    like any other, so the extra items are dropped rather than rejected.

    !!! warning
        This is deliberately more permissive than
        `HasLength` in `bagof-validators`, which
        refuses a mismatch in either direction. Validate first if you
        need a long sequence to be an error rather than a trim.

    !!! example
        ```pycon
        >>> from bagof.converters.collections import ToLength
        >>> to_pair = ToLength(2, tx.List[int])
        >>> to_pair(["1", "2", "3"])
        [1, 2]
        >>> to_pair(["1"])
        ValueConversionError: Expected sequence of length 2, got 1.
        ```
    """

    def __init__(
        self,
        length: int,
        hint: tx.Any = UNSET,
        compose: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        length : int
            The expected length of the sequence.
        hint : Any, optional
            The type hint to convert to.
        compose : bool
            Whether to compose this converter with others, when they are
            found in [`Annotated`][typing.Annotated] metadata.
        """
        super().__init__(hint, compose)
        self.length = length

    def __call__(self, value: tx.Any) -> ITERABLE:
        """Convert the value to a sequence, trimmed to the length."""
        value = super().__call__(value)
        # Trim before checking: a longer input is coerced, not refused.
        value = value[:self.length]
        if len(value) != self.length:
            raise self.value_error(
                value,
                f"Expected sequence of length {self.length}, "
                f"got {len(value)}.",
            )
        return value  # type: ignore[return-value]

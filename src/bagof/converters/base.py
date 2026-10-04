"""Base class for all converters."""

__all__ = [
    "Converter",
    "ConverterRegistry",
    "register_converter",
    "get_converter",
    "get_converter_class",
    "wrap_converter",
]

# stdlib
import importlib
import importlib.util
import re
import sys

# dependencies
import typing_extensions as tx  # noqa: I001

# bags
from bagof.core.magic import (
    UNSET,
    MagicHint,
    get_from_registry,
    safe_isinstance,
    safe_issubclass,
)
from bagof.hints.typevars.co import T

# locals
from .exceptions import (
    ConversionError,
    TypeConversionError,
    ValueConversionError,
)

# typing
ClassDecorator: tx.TypeAlias = tx.Callable[[T], T]
"""A class decorator (that takes a class and returns a class)."""

ConverterRegistry = tx.Dict[tx.Hashable, tx.Type["Converter"]]
"""A registry of converters, mapping type hints to converter classes."""

FROM = tx.TypeVar("FROM")
"""TypeVar for converter input types."""

TO = tx.TypeVar("TO")
"""TypeVar for converter output types."""

# constants
CONVERTERS: ConverterRegistry = {}
"""The global registry of converters."""

_NAME_KEY = re.compile(r"^[\w.]+:[\w.]+$")
"""Pattern of a lazy, qualified-name registry key (`"module:qualname"`)."""

_PENDING: tx.List[tx.Tuple[ConverterRegistry, str, tx.Type["Converter"]]] = []
"""
Name-keyed registrations whose module has not been imported yet, as
`(registry, "module:qualname", converter)`. They are moved into their
registry -- keyed by the real object -- once the module is in
[`sys.modules`][]. A hint or value of a type from a module that was never
imported cannot reach the registry, so nothing is lost by waiting.
"""


def _has_module(name: str) -> bool:
    """Whether a module can be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover
        return False


def _import_name(key: str) -> tx.Any:
    """Import the object named by `"module"` or `"module:qualname"`."""
    module, _, qualname = key.partition(":")
    obj = importlib.import_module(module)
    for attr in filter(None, qualname.split(".")):
        obj = getattr(obj, attr)
    return obj


class _lazy:
    """
    Class attribute holding an object that is only imported on first access,
    so that defining a converter for an optional library does not import it.

    !!! example
        ```python
        class ToDaskArray(ArrayConverter):
            DEFAULT = _lazy("dask.array:Array")
        ```
    """

    def __init__(self, key: str) -> None:
        self.key = key

    def __get__(self, obj: tx.Any, owner: tx.Any = None) -> tx.Any:
        if "value" not in self.__dict__:
            self.value = _import_name(self.key)
        return self.value


def _resolve_pending() -> None:
    """Register the pending name keys whose module is now imported."""
    for entry in list(_PENDING):
        registry, key, cls = entry
        module, _, qualname = key.partition(":")
        obj = sys.modules.get(module)
        if obj is None:
            continue
        try:
            for attr in qualname.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            # Still initialising (or no such name): try again next time.
            continue
        _PENDING.remove(entry)
        registry[obj] = cls


class ConverterMetaclass(type(MagicHint)):
    """Metaclass for all converters."""

    def __new__(
        metacls,
        name: str,
        bases: tx.Tuple[type, ...],
        namespace: tx.Mapping[str, tx.Any],
        **kwargs: tx.Any,
    ) -> tx.Self:
        register = kwargs.pop("register", UNSET)
        cls = super().__new__(metacls, name, bases, namespace, **kwargs)
        if register is not UNSET:
            if register is True:
                register = (cls.DEFAULT,)
            if not isinstance(register, tuple):
                register = (register,)
            Converter.register(cls, *register)
        return cls


class Converter(
    MagicHint[TO], tx.Generic[TO, FROM], metaclass=ConverterMetaclass
):
    """
    Base class for magic converters.

    Converters take a value and return a converted version of it.
    They are registered in a global registry and looked up by type hint.

    !!! note
        Failed conversions raise a [`ConversionError`][]. Its concrete
        subclasses -- [`ValueConversionError`][] (also a [`ValueError`][])
        and [`TypeConversionError`][] (also a [`TypeError`][]) -- inherit
        the matching builtin, so callers that already catch `ValueError`
        or `TypeError` still catch conversion failures.
    """

    DEFAULT = tx.Any

    def __init__(self, hint: tx.Any = UNSET, compose: bool = False) -> None:
        """
        Parameters
        ----------
        hint : Any, optional
            The type hint to use for this magic object.
            If not provided, the default hint for the class is used.
        compose : bool
            Whether to compose this converter with others, when they are
            found in [`Annotated`][typing.Annotated] metadata.
        """
        super().__init__(hint)
        self.compose = compose

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """
        Return a type hint describing valid inputs for this converter.

        Parameters
        ----------
        __reentrant : tuple
            Used internally to avoid infinite recursion.

        Returns
        -------
        Any
            A type hint for valid input values.
        """
        return tx.Any

    def __call__(self, value: FROM) -> TO:
        """
        Convert the given value.

        Parameters
        ----------
        value : FROM
            The value to convert.

        Returns
        -------
        TO
            The converted value.

        Raises
        ------
        ConversionError
            If the value cannot be converted.
        """
        if not safe_isinstance(value, self.origin):
            value = self._wrap_converter(self.origin)(value)
        return tx.cast(TO, value)

    def error(
        self, value: tx.Any, message: tx.Optional[str] = None, **kwargs: tx.Any
    ) -> ConversionError:
        """Return a [`ConversionError`][] with the given value and message."""
        type = kwargs.pop("type", ConversionError)
        type = {
            "value": ValueConversionError,
            "type": TypeConversionError,
        }.get(type, type)
        kwargs.setdefault("this", self)
        kwargs.setdefault("value", value)
        if message is None:
            message = "Invalid value."
        return type(message, **kwargs)

    def type_error(
        self, value: tx.Any, message: tx.Optional[str] = None
    ) -> TypeConversionError:
        """Return a [`TypeConversionError`][] with the given value."""
        if message is None:
            message = f"Invalid value type: {type(value)}"
        return self.error(value, message, type=TypeConversionError)

    def value_error(
        self, value: tx.Any, message: tx.Optional[str] = None
    ) -> ValueConversionError:
        """Return a [`ValueConversionError`][] with the given value."""
        if message is None:
            message = "Invalid value."
        return self.error(value, message, type=ValueConversionError)

    def _wrap_converter(self, converter: tx.Callable) -> tx.Callable:
        """
        Wrap a converter to catch errors and raise a
        [`ConversionError`][] instead.
        """
        return _trywrap_converter(converter, self.value_error)

    @tx.overload
    @staticmethod
    def register(
        converter: tx.Type["Converter"],
        *hints: tx.Unpack[tx.Tuple[tx.Any, ...]],
        registry: ConverterRegistry = ...,
    ) -> tx.Type["Converter"]:
        ...

    @tx.overload
    @staticmethod
    def register(
        *hints: tx.Unpack[tx.Tuple[tx.Any, ...]],
        registry: ConverterRegistry = ...,
    ) -> ClassDecorator:
        ...

    @staticmethod
    def register(  # type: ignore[misc]
        *hints: tx.Any,
        registry: ConverterRegistry = CONVERTERS,
    ) -> tx.Any:
        """
        Register a converter class for one or more type hints.

        Can be used as a decorator or called directly:

        !!! example
            ```python
            @Converter.register(int)
            class ToInt(Converter[int, str]):
                def __call__(self, value: str) -> int:
                    return int(value)
            ```

        Parameters
        ----------
        *hints
            One or more type hints to register the converter class for.
            A string `"module:qualname"` (e.g. `"dask.array:Array"`)
            registers the object of that name lazily, without importing
            `module`: the key is resolved once `module` has been imported.
        registry : ConverterRegistry
            The registry to register the converter class in.
            Defaults to the global registry.
        """
        if hints and safe_issubclass(hints[0], Converter):
            converter, *hints = hints
            return Converter.register(*hints, registry=registry)(converter)

        def decorator(cls: tx.Type[Converter]) -> tx.Type[Converter]:
            hints_ = hints or (cls.DEFAULT,)
            # Settle the resolvable name keys first, so that the later
            # registration still wins, whichever kind of key each one is.
            _resolve_pending()
            for hint in hints_:
                if isinstance(hint, str) and _NAME_KEY.match(hint):
                    _PENDING[:] = [
                        entry for entry in _PENDING
                        if entry[0] is not registry or entry[1] != hint
                    ]
                    _PENDING.append((registry, hint, cls))
                else:
                    registry[hint] = cls
            _resolve_pending()
            return cls

        return decorator

    @staticmethod
    def get(
        hint: tx.Any,
        registry: ConverterRegistry = CONVERTERS,
        fallback: tx.Optional[tx.Type["Converter"]] = UNSET,
    ) -> tx.Optional["Converter"]:
        """
        Get the best-matching converter for a given type hint.

        !!! example
            ```pycon
            >>> from bagof.converters import get_converter
            >>> convert = get_converter(list[int])
            >>> convert(["1", "2", "3"])
            [1, 2, 3]
            >>> get_converter(dict[str, int])({"a": "1", "b": "2"})
            {'a': 1, 'b': 2}
            ```

        Parameters
        ----------
        hint : Any
            The type hint for which to get a converter.
        registry : ConverterRegistry
            The registry to look up the converter in.
            Defaults to the global registry.
        fallback : Optional[Type[Converter]]
            The fallback converter class to use if no matching converter
            is found. Defaults to [`Converter`][].
            Pass `None` explicitly to get `None` instead of a fallback.

        Returns
        -------
        Optional[Converter]
            The best-matching converter for the given type hint, or `None`
            if no matching converter is found and no fallback is provided.
        """
        cls = Converter.get_class(hint, registry, fallback)
        if cls is None:
            return None
        return cls(hint)

    @staticmethod
    def get_class(
        hint: tx.Any,
        registry: ConverterRegistry = CONVERTERS,
        fallback: tx.Optional[tx.Type["Converter"]] = UNSET,
    ) -> tx.Optional[tx.Type["Converter"]]:
        """
        Get the best-matching converter class for a given type hint.

        Parameters
        ----------
        hint : Any
            The type hint for which to get a converter.
        registry : ConverterRegistry
            The registry to look up the converter in.
            Defaults to the global registry.
        fallback : Optional[Type[Converter]]
            The fallback converter class to use if no matching converter
            is found. Defaults to [`Converter`][].
            Pass `None` explicitly to get `None` instead of a fallback.

        Returns
        -------
        Optional[Type[Converter]]
            The best-matching converter class for the given type hint,
            or `None` if no matching converter is found and no fallback
            is provided.
        """
        if fallback is UNSET:
            fallback = Converter
        if _PENDING:
            _resolve_pending()
        return get_from_registry(hint, registry) or fallback


register_converter = Converter.register
"""Backward-compatible alias for [`Converter.register`][]."""

get_converter = Converter.get
"""Backward-compatible alias for [`Converter.get`][]."""

get_converter_class = Converter.get_class
"""Backward-compatible alias for [`Converter.get_class`][]."""


def wrap_converter(
    converter: "Converter",
    TO: tx.Any = UNSET,
    FROM: tx.Any = UNSET,
) -> tx.Callable:
    """
    Wrap a converter so that it has the correct input and output annotations.

    Parameters
    ----------
    converter : Converter
        The converter to wrap.
    TO : Any, optional
        The output type hint. Defaults to `converter.hint`.
    FROM : Any, optional
        The input type hint. Defaults to `converter.like()`.

    Returns
    -------
    Callable
        A callable that wraps `converter`, annotated with `FROM` as its
        parameter type and `TO` as its return type.
    """
    if TO is UNSET:
        TO = converter.hint

    if FROM is UNSET:
        to_converter = converter
        if TO != converter.hint:
            to_converter = Converter.get(TO)
        FROM = to_converter.like()

    def convert(value: FROM) -> TO:
        return converter(value)

    return convert


def _process_reentrant(inp: tx.Any, reentrant: tuple = ()) -> tuple:
    """
    Process a reentrant type hint to avoid infinite recursion.

    If the input hint is already in the reentrant tuple, returns an empty
    tuple (falsy). Otherwise, appends the hint to the reentrant tuple and
    returns the updated tuple (truthy).
    """
    if inp in reentrant:
        return ()
    reentrant += (inp,)
    return reentrant


def _trywrap_converter(
    converter: tx.Callable, error: tx.Any
) -> tx.Callable:
    """
    Wrap a converter so that a plain [`TypeError`][] or [`ValueError`][]
    raised inside it surfaces as a [`ConversionError`][], with the
    original attached as its cause.

    A [`ConversionError`][] raised by the wrapped converter passes
    through unchanged.
    """
    def wrapped(value: tx.Any) -> tx.Any:
        try:
            return converter(value)
        except ConversionError:
            # Already the right kind of error: re-raise it untouched, so
            # its specific type, message and `causes` survive. `TypeError`
            # and `ValueError` are the base classes of
            # `TypeConversionError` and `ValueConversionError`, so without
            # this the `except` below downgrades every real failure to a
            # generic "Invalid value."
            raise
        except (TypeError, ValueError) as e:
            _error = error
            if callable(_error):
                _error = _error(value)
            raise _error from e
    return wrapped

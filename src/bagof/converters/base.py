"""Base class for all converters."""

__all__ = [
    "Converter",
    "ConverterRegistry",
    "register_converter",
    "get_converter",
    "get_converter_class",
    "wrap_converter",
]

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
    """

    DEFAULT = tx.Any

    def __init__(self, hint: tx.Any = UNSET, compose: bool = False) -> None:
        """
        Parameters
        ----------
        hint
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
        *hints: tx.Unpack[tx.Tuple[tx.Any]],
        registry: ConverterRegistry = ...,
    ) -> tx.Type["Converter"]:
        ...

    @tx.overload
    @staticmethod
    def register(
        *hints: tx.Unpack[tx.Tuple[tx.Any]],
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
        registry : ConverterRegistry
            The registry to register the converter class in.
            Defaults to the global registry.
        """
        if hints and safe_issubclass(hints[0], Converter):
            converter, *hints = hints
            return Converter.register(*hints, registry=registry)(converter)

        def decorator(cls: tx.Type[Converter]) -> tx.Type[Converter]:
            hints_ = hints or (cls.DEFAULT,)
            for hint in hints_:
                registry[hint] = cls
            return cls

        return decorator

    @staticmethod
    def get(
        hint: tx.Any,
        registry: ConverterRegistry = CONVERTERS,
        fallback: tx.Optional[tx.Type["Converter"]] = None,
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
        hint
            The type hint for which to get a converter.
        registry : ConverterRegistry
            The registry to look up the converter in.
            Defaults to the global registry.
        fallback : Optional[Type[Converter]]
            The fallback converter class to use if no matching converter
            is found.

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
        fallback: tx.Optional[tx.Type["Converter"]] = None,
    ) -> tx.Optional[tx.Type["Converter"]]:
        """
        Get the best-matching converter class for a given type hint.

        Parameters
        ----------
        hint
            The type hint for which to get a converter.
        registry : ConverterRegistry
            The registry to look up the converter in.
            Defaults to the global registry.
        fallback : Optional[Type[Converter]]
            The fallback converter class to use if no matching converter
            is found.

        Returns
        -------
        Optional[Type[Converter]]
            The best-matching converter class for the given type hint,
            or `None` if no matching converter is found and no fallback
            is provided.
        """
        return get_from_registry(hint, registry) or fallback


register_converter = Converter.register
"""Backward-compatible alias for [`Converter.register`][]"""

get_converter = Converter.get
"""Backward-compatible alias for [`Converter.get`][]"""

get_converter_class = Converter.get_class
"""Backward-compatible alias for [`Converter.get_class`][]"""


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
    TO : Any
        The output type hint. Defaults to `converter.hint`.
    FROM : Any
        The input type hint. Defaults to `converter.like()`.
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
    """Wrap a converter to catch errors and raise a ConversionError instead."""
    def wrapped(value: tx.Any) -> tx.Any:
        try:
            return converter(value)
        except (TypeError, ValueError) as e:
            _error = error
            if callable(_error):
                _error = _error(value)
            raise _error from e
    return wrapped

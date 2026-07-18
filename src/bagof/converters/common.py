"""Common converters (any, union, etc.)"""

__all__ = [
    "ToAny",
    "ToNone",
    "ToType",
    "ToUnion",
    "ToLiteral",
    "ToTypeVar",
    "ToAnnotated",
]

# dependencies
import typing_extensions as tx

# bags
from bagof.core.magic import (
    UNSET,
    MultipleCauses,
    get_args_uw,
    issubhint,
    safe_get_args,
    safe_isinstance,
    safe_issubclass,
    unwrap,
)
from bagof.hints.typevars.co import NONE, NoneType

# locals
from ._compat import UnionType
from .base import (
    FROM,
    TO,
    ClassDecorator,
    Converter,
    ConverterRegistry,
    _process_reentrant,
)
from .exceptions import TypeConversionError

# --- Any --------------------------------------------------------------


class ToAny(Converter[TO, FROM], register=tx.Any):
    """Converter for [`Any`][typing.Any] (no-op, returns the value as-is)."""

    BOUND = DEFAULT = tx.Any

    def __call__(self, value: FROM) -> TO:
        """Return the value unchanged."""
        return value  # type: ignore[return-value]


# --- None -------------------------------------------------------------


class ToNone(Converter[NONE, tx.Any], register=NoneType):
    """Converter for [`None`][]."""

    BOUND = DEFAULT = NoneType

    def __call__(self, value: tx.Any) -> NONE:
        """Return the value if it is None, otherwise raise a TypeError."""
        if value is not None:
            raise self.type_error(value, "Value is not None")
        return value  # type: ignore[return-value]


# --- Type -------------------------------------------------------------


class ToType(Converter[TO, FROM], register=type):
    """
    Converter for [`type`][] and [`Type[T]`][typing.Type].

    This is a *validating* converter: it does not coerce, it checks that the
    value is a class (and, for `Type[T]`, a subclass of `T`).
    """

    DEFAULT = type

    def __call__(self, value: FROM) -> TO:
        """Return the value if it is a (sufficiently specific) type."""
        if not isinstance(value, type):
            raise self.type_error(value, "Value is not a type.")
        args = self.args
        if args and not safe_issubclass(value, args[0]):
            raise self.value_error(
                value, f"Value is not a subclass of {args[0]!r}."
            )
        return value  # type: ignore[return-value]


# --- Union ------------------------------------------------------------


class ToUnion(Converter[TO, FROM], register=(tx.Union, UnionType)):
    """
    Converter for [`Union`][typing.Union].

    !!! example
        Branches are tried in order; the first that converts wins:

        ```pycon
        >>> from bagof.converters import get_converter
        >>> convert = get_converter(int | str)
        >>> convert("5")
        5
        >>> convert("x")
        'x'
        ```
    """

    BOUND = DEFAULT = tx.Union

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.args:
            raise TypeError(
                f"Hint cannot be an empty or general union: {self.hint}"
            )

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Return the union of the `like` hints for each union branch."""
        return _like_union(self.unwrapped, __reentrant)

    def __call__(self, value: FROM) -> TO:
        """Try each branch of the union in order; raise if none succeeds."""
        return _to_union(
            value, self.unwrapped, self._notinunion_error(value)
        )

    def _notinunion_error(self, value: tx.Any) -> TypeConversionError:
        return self.type_error(
            value,
            "Value not compatible with any of the union types",
        )


def _like_union(hint: tx.Any, __reentrant: tuple = ()) -> tx.Any:
    __reentrant = _process_reentrant(hint, __reentrant)
    if not __reentrant:
        if hint in (tx.Union, UnionType):
            return UNSET
        return hint

    if not issubhint(hint, tx.Union):
        raise TypeError(f"Hint {hint} is not a Union type")

    args = get_args_uw(hint)
    args = tuple(
        Converter.get(arg).like(__reentrant)
        for arg in args
    )
    args = tuple(arg for arg in args if arg is not UNSET)

    # Only keep the more specific hints (remove super hints)
    filtered_args: list = []
    for arg in args:
        for filtered_arg in filtered_args:
            if issubhint(arg, filtered_arg):
                continue
            if issubhint(filtered_arg, arg):
                filtered_args.remove(filtered_arg)
                break
        filtered_args.append(arg)

    return tx.Union[tuple(filtered_args)] if filtered_args else tx.Never


def _to_union(
    value: tx.Any, hint: tx.Any, type_error: TypeConversionError
) -> tx.Any:
    args = get_args_uw(hint)

    # short-circuit for NoneType
    if value is None and NoneType in args:
        return None

    errors = []
    for arg in args:
        try:
            converter = Converter.get(arg)
            return converter(value)
        except (TypeError, ValueError) as e:
            errors.append(e)
            continue

    raise type_error from MultipleCauses(errors)


# --- Literal ----------------------------------------------------------


class ToLiteral(Converter[TO, FROM], register=tx.Literal):
    """Converter for [`Literal`][typing.Literal]."""

    BOUND = DEFAULT = tx.Literal

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Return the literal hint itself."""
        return self.hint

    def __call__(self, value: FROM) -> TO:
        """Return the value if it is one of the literals; raise otherwise."""
        if value not in self.args:
            raise self.value_error(
                value, "Value is not compatible with any of the literals."
            )
        return value  # type: ignore[return-value]


# --- TypeVar ----------------------------------------------------------


class ToTypeVar(Converter[TO, FROM], register=tx.TypeVar):
    """Converter for [`TypeVar`][typing.TypeVar]."""

    BOUND = DEFAULT = tx.TypeVar("T")

    # `unwrapped` resolves the typevar (see `MagicHint.UNWRAP`), so this
    # re-dispatches to the converter registered for the bound itself.

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Return the `like` hint for the unwrapped TypeVar bound."""
        __reentrant = _process_reentrant(self.hint, __reentrant)
        if not __reentrant:
            return self.hint
        return Converter.get(self.unwrapped).like(__reentrant)

    def __call__(self, value: FROM) -> TO:
        """Delegate to the converter for the unwrapped TypeVar."""
        result = Converter.get(self.unwrapped)(value)
        return result  # type: ignore[return-value]


# --- Annotated --------------------------------------------------------


class ToAnnotated(Converter[TO, FROM], register=tx.Annotated):
    """
    Converter for [`Annotated`][typing.Annotated].

    !!! note
        Annotated converters look for converters in the metadata of an
        annotated type hint and apply them in sequence (if they are
        composable).
    """

    _REGISTRY: ConverterRegistry = {}

    @classmethod
    def register(  # type: ignore[override]
        cls, *hints: tx.Unpack[tx.Tuple[tx.Any]]
    ) -> ClassDecorator:
        """
        Register a converter class for use as [`Annotated`][typing.Annotated]
        metadata.

        !!! example
            ```python
            @ToAnnotated.register(re.Pattern)
            class ToRegexMatch(ToString):
                ...
            ```
        """
        def decorator(
            converter_cls: tx.Type[Converter],
        ) -> tx.Type[Converter]:
            for hint in hints:
                cls._REGISTRY[hint] = converter_cls
            return converter_cls

        return decorator

    @classmethod
    def _get_converter(
        cls, hint: tx.Any
    ) -> tx.Optional["Converter"]:
        # First try a direct registry lookup (works for types/hints).
        converter = Converter.get(hint, registry=cls._REGISTRY, fallback=None)
        if converter is not None:
            return converter
        # If hint is an instance (e.g. re.compile(r"\d+")), look up its type
        # (e.g. re.Pattern) and instantiate the converter with the instance as
        # the first positional argument (e.g. ToRegexMatch(pattern)).
        if not isinstance(hint, type):
            converter_cls = Converter.get_class(
                type(hint), registry=cls._REGISTRY, fallback=None
            )
            if converter_cls is not None:
                return converter_cls(hint)
        return None

    @property
    def converters(self) -> tx.Tuple[Converter, ...]:
        """The chain of converters derived from the Annotated metadata."""
        if getattr(self, "_converters", None) is None:
            self._converters = self._get_converters()
        return self._converters

    def _get_converters(self) -> tx.Tuple[Converter, ...]:
        unwrapped = unwrap(self.hint)
        converters = []
        for arg in safe_get_args(self.hint):
            if safe_issubclass(arg, Converter):
                arg = arg(unwrapped)
            if not safe_isinstance(arg, Converter):
                arg = self._get_converter(arg)
            if safe_isinstance(arg, Converter):
                if getattr(arg, "compose", False):
                    converters.append(arg)
                else:
                    converters = [arg]

        if not converters or getattr(converters[0], "compose", False):
            converters.insert(0, Converter.get(unwrapped))

        return tuple(converters)

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Return the `like` hint from the first converter in the chain."""
        __reentrant = _process_reentrant(self.hint, __reentrant)
        if not __reentrant:
            return self.hint
        return self.converters[0].like(__reentrant)

    def __call__(self, value: FROM) -> TO:
        """Apply each converter in the chain in sequence."""
        for converter in self.converters:
            # NOTE: do not catch and rethrow here — helps with legibility.
            value = converter(value)
        return value  # type: ignore[return-value]

"""Common converters (any, union, etc.)."""

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
    ishintstance,
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
from .exceptions import ConversionError, TypeConversionError

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

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Accept only `None`."""
        return NoneType

    def __call__(self, value: tx.Any) -> NONE:
        """Return the value if it is None, otherwise raise a TypeError."""
        if value is not None:
            raise self.type_error(value, "Value is not None")
        return value  # type: ignore[return-value]


# --- Type -------------------------------------------------------------


class ToType(Converter[TO, FROM], register=type):
    """
    Converter for [`type`][] and [`Type[T]`][typing.Type].

    !!! note
        This is a *validating* converter: it does not coerce, it checks
        that the value is a class (and, for `Type[T]`, a subclass
        of `T`).
    """

    DEFAULT = type

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Accept a class."""
        return type

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

    A value that already matches one of the branches is returned
    unchanged. Otherwise the branches are tried in order, and the first
    that converts wins.

    !!! example
        ```pycon
        >>> from bagof.converters import get_converter
        >>> convert = get_converter(int | str)
        >>> convert("5")        # already a str
        '5'
        >>> convert(5)          # already an int
        5
        >>> convert(b"5")       # matches neither; the first that converts
        5
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

    # A branch that accepts anything must not swallow the others. It is a
    # super hint of every one of them, so the redundancy filter below
    # would collapse the whole union to `Any` -- which is true but
    # useless, and is what `wrap_converter` would then annotate with.
    specific = tuple(arg for arg in args if arg is not tx.Any)
    if not specific:
        return tx.Any if args else tx.Never
    args = specific

    # Only keep the more specific hints (remove super hints)
    filtered_args: list = []
    for arg in args:
        if any(issubhint(arg, kept) for kept in filtered_args):
            # `arg` is redundant: an existing entry already covers it
            continue
        filtered_args = [
            kept for kept in filtered_args if not issubhint(kept, arg)
        ]
        filtered_args.append(arg)

    return tx.Union[tuple(filtered_args)] if filtered_args else tx.Never


def _to_union(
    value: tx.Any, hint: tx.Any, type_error: TypeConversionError
) -> tx.Any:
    args = get_args_uw(hint)

    # A value that already satisfies one of the branches is returned
    # unchanged. Without this pass the result depends on how the union was
    # spelled -- `Union[int, str]("5")` gave `5` while `Union[str, int]
    # ("5")` gave `"5"`, even though the two hints denote the same type --
    # and a perfectly valid `str` came back coerced to an `int`.
    #
    # `None` is the special case this generalises: it was already
    # short-circuited here, because no amount of branch order should turn
    # `None` into something else.
    for arg in args:
        if ishintstance(value, arg):
            return value

    errors = []
    for arg in args:
        converter = Converter.get(arg)
        try:
            return converter(value)
        except ConversionError as e:
            # Only a conversion failure means "this branch did not match".
            # Catching the builtins here as well swallowed genuine bugs in
            # a converter's own code and reported them as "no branch
            # matched"; a branch that raises a plain `TypeError` is a
            # branch that is broken, and that should surface.
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
        """Return the matching literal; raise if the value is not one."""
        for arg in self.args:
            # Match by type as well as value, as PEP 586 specifies: `True`
            # is not a valid `Literal[1]` even though `True == 1`. And
            # return the literal itself rather than the input, so the
            # result really is of the hinted type -- returning `1.0` for
            # `Literal[1]` would be a conversion that converts nothing.
            if ishintstance(value, tx.Literal[arg]):
                return arg  # type: ignore[return-value]
        raise self.value_error(
            value, "Value is not compatible with any of the literals."
        )


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
    def register_metadata(
        cls, *hints: tx.Unpack[tx.Tuple[tx.Any, ...]]
    ) -> ClassDecorator:
        """
        Register a converter class for use as [`Annotated`][typing.Annotated]
        metadata.

        Distinct from [`Converter.register`][], which registers a
        converter for a *type hint* in the global registry; this one
        registers it for a piece of `Annotated` **metadata**.

        !!! example
            ```python
            @ToAnnotated.register_metadata(re.Pattern)
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

    # Deprecated alias. `register` means "register for a type hint"
    # everywhere else, and a bare `@ToAnnotated.register` used to
    # silently register the decorated class as a metadata *key*.
    register = register_metadata

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
                # Bind by keyword. Not every converter takes `hint` first
                # -- `ToLength(length, ...)` and `ToRegexMatch(pattern,
                # ...)` take their own configuration there -- and passing
                # positionally silently bound the annotated type as that
                # argument. By keyword, such a class raises a `TypeError`
                # naming what it is missing, which points at the real fix:
                # write an instance rather than the bare class.
                arg = arg(hint=unwrapped)
            if not safe_isinstance(arg, Converter):
                arg = self._get_converter(arg)
            if safe_isinstance(arg, Converter):
                if not arg.has_explicit_hint:
                    # A metadata converter written without a hint of its
                    # own converts to the annotated type. Without this, a
                    # non-composable one - which replaces the base
                    # converter - was left converting to its class
                    # `DEFAULT` instead.
                    arg = arg.rebind(unwrapped)
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

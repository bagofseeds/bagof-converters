"""Converter for TypedDict types."""

__all__ = ["ToTypedDict"]

# dependencies
import typing_extensions as tx

# bags
from bagof.core.magic import safe_get_args, safe_get_origin, unwrap

# locals
from .base import Converter
from .collections import ToMapping

_REQUIREDNESS = (tx.Required, tx.NotRequired)


def _strip_requiredness(hint: tx.Any) -> tx.Any:
    """
    Remove a `Required` / `NotRequired` wrapper, inside or outside
    `Annotated`.

    Both nestings are legal (PEP 655) and typing keeps whichever was
    written, so `Annotated[NotRequired[int], "m"]` has to be unwrapped
    from the inside -- keeping the metadata, which may carry converters.
    """
    inner = unwrap(hint, _REQUIREDNESS)
    if inner is not hint:
        return _strip_requiredness(inner)
    if safe_get_origin(hint) is tx.Annotated:
        args = safe_get_args(hint)
        if args:
            base, meta = args[0], args[1:]
            stripped = unwrap(base, _REQUIREDNESS)
            if stripped is not base:
                return tx.Annotated[(stripped,) + meta]
    return hint


class ToTypedDict(ToMapping, register=tx.TypedDict):
    """
    Converter for [`TypedDict`][typing.TypedDict] subclasses.

    Each key present in the input is converted through the converter for
    its declared type. Optional keys (from ``total=False`` or
    [`NotRequired`][typing.NotRequired]) may be absent; a missing
    **required** key is an error -- building a value for it is a
    factory's job, not a converter's.

    !!! example
        ```pycon
        >>> import typing_extensions as tx
        >>> from bagof.converters import get_converter
        >>> class Movie(tx.TypedDict):
        ...     title: str
        ...     year: int
        >>> get_converter(Movie)({"title": "Alien", "year": "1979"})
        {'title': 'Alien', 'year': 1979}
        ```
    """

    DEFAULT = tx.TypedDict
    FALLBACK = dict

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Accept any mapping."""
        return tx.Mapping[str, tx.Any]

    def __call__(self, value: tx.Any) -> tx.Any:
        """Convert each present key through its declared type."""
        # Check the container first. A TypedDict is a `dict` at runtime and
        # cannot be instance-checked, so validate against `dict` itself.
        value = ToMapping(dict)(value)

        cls = self.origin
        annotations = tx.get_type_hints(cls, include_extras=True)
        required = getattr(cls, "__required_keys__", None)
        if required is None:  # pragma: no cover - every TypedDict has it
            required = frozenset(annotations)
        closed = getattr(cls, "__closed__", False)
        extra_items = getattr(cls, "__extra_items__", tx.Any)
        if extra_items is getattr(tx, "NoExtraItems", tx.Any):
            extra_items = tx.Any

        result = {}
        for key, hint in annotations.items():
            if key not in value:
                if key in required:
                    raise self.value_error(
                        value, f"Missing required key {key!r}."
                    )
                continue
            converter = Converter.get(_strip_requiredness(hint))
            result[key] = self._wrap_converter(converter)(value[key])

        for key, item in value.items():
            if key in annotations:
                continue
            if closed:
                raise self.value_error(value, f"Unexpected key {key!r}.")
            converter = Converter.get(extra_items)
            result[key] = self._wrap_converter(converter)(item)

        return result

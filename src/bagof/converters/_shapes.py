"""
Shape hints of array types.

An array hint such as `ndarray[Tuple[Literal[3], int], dtype[float64]]`
carries its shape as a tuple hint. This module reads that tuple into a
[`ShapeSpec`][], which says how many dimensions an array may have and
which sizes each of them may take:

* `int` (or `Any`, a `TypeVar`, ...) accepts any size;
* `Literal[3]` accepts a size of 3, and `Literal[1, 3]` a size of 1 or 3;
* `Tuple[X, ...]` accepts any number of dimensions, each matching `X`;
* `Tuple[X, Unpack[Tuple[Y, ...]], Z]` (or `tuple[X, *tuple[Y, ...], Z]`)
  accepts at least two dimensions: the first matches `X`, the last
  matches `Z`, and any in between match `Y`. An unpacked `TypeVarTuple`
  works as `Unpack[Tuple[int, ...]]`.

!!! note
    This module only depends on `bagof.core.magic` and
    `typing_extensions`, and is shared verbatim by the converters,
    validators and factories bags.
"""

__all__ = ["ShapeSpec", "shape_spec"]

# dependencies
import typing_extensions as tx

# bags
from bagof.core.magic import safe_get_args, safe_get_origin, unwrap

DimSpec = tx.Optional[tx.Tuple[int, ...]]
"""The sizes a dimension may take, or `None` for any size."""


class ShapeSpec:
    """
    The shapes an array hint accepts.

    Attributes
    ----------
    head : tuple[DimSpec, ...]
        The leading dimensions.
    star : DimSpec
        The dimensions between `head` and `tail`, if `variadic`.
    tail : tuple[DimSpec, ...]
        The trailing dimensions.
    variadic : bool
        Whether any number of dimensions may sit between `head` and
        `tail`. Otherwise, the array has exactly `len(head) + len(tail)`
        dimensions.
    """

    def __init__(
        self,
        head: tx.Tuple[DimSpec, ...] = (),
        star: DimSpec = None,
        tail: tx.Tuple[DimSpec, ...] = (),
        variadic: bool = False,
    ) -> None:
        self.head = tuple(head)
        self.star = star if variadic else None
        self.tail = tuple(tail)
        self.variadic = variadic

    @property
    def ndim(self) -> int:
        """The smallest number of dimensions accepted."""
        return len(self.head) + len(self.tail)

    def accepts_ndim(self, ndim: int) -> bool:
        """Whether an array with `ndim` dimensions may match."""
        if self.variadic:
            return ndim >= self.ndim
        return ndim == self.ndim

    def align(self, ndim: int) -> tx.Tuple[DimSpec, ...]:
        """
        The size constraint of each dimension of an array with `ndim`
        dimensions, which [`accepts_ndim`][] must accept.
        """
        middle = (self.star,) * (ndim - self.ndim)
        return self.head + middle + self.tail

    def check(self, shape: tx.Tuple[int, ...]) -> tx.Optional[str]:
        """Why `shape` does not match, or `None` if it does."""
        if not self.accepts_ndim(len(shape)):
            return (
                f"Expected an array of shape {self}, "
                f"got one with {len(shape)} dimension(s) {tuple(shape)}."
            )
        for size, sizes in zip(shape, self.align(len(shape))):
            if not _size_ok(size, sizes):
                return (
                    f"Expected an array of shape {self}, "
                    f"got one of shape {tuple(shape)}."
                )
        return None

    def broadcast_shape(
        self, shape: tx.Tuple[int, ...]
    ) -> tx.Tuple[int, ...]:
        """
        The shape that `shape` broadcasts to, following numpy's rules: it
        is first left-padded with singleton dimensions, up to the smallest
        number of dimensions accepted, and each singleton dimension
        constrained to a single size is then expanded to that size.

        Raises
        ------
        ValueError
            If `shape` cannot be broadcast to a matching shape.
        """
        original = tuple(shape)
        shape = (1,) * max(0, self.ndim - len(shape)) + original
        if not self.accepts_ndim(len(shape)):
            raise ValueError(
                f"Cannot broadcast an array of shape {original} to "
                f"{self}: too many dimensions."
            )
        target = []
        for size, sizes in zip(shape, self.align(len(shape))):
            if _size_ok(size, sizes):
                target.append(size)
            elif size == 1 and len(sizes) == 1:
                target.append(sizes[0])
            else:
                raise ValueError(
                    f"Cannot broadcast an array of shape {original} to "
                    f"{self}."
                )
        return tuple(target)

    def default_shape(self) -> tx.Tuple[int, ...]:
        """
        The smallest shape that matches: no dimension in the variadic
        part, the first size of each constrained dimension, and a size
        of 0 for each free one.
        """
        return tuple(0 if s is None else s[0] for s in self.head + self.tail)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShapeSpec):
            return NotImplemented
        return (
            self.head == other.head
            and self.star == other.star
            and self.tail == other.tail
            and self.variadic == other.variadic
        )

    def __hash__(self) -> int:
        return hash((self.head, self.star, self.tail, self.variadic))

    def __repr__(self) -> str:
        dims = [_dim_repr(s) for s in self.head]
        if self.variadic:
            dims.append(f"*{_dim_repr(self.star)}")
        dims += [_dim_repr(s) for s in self.tail]
        if len(dims) == 1 and not self.variadic:
            return f"({dims[0]},)"
        return f"({', '.join(dims)})"


def shape_spec(hint: tx.Any) -> tx.Optional[ShapeSpec]:
    """
    Read a shape hint (the first argument of an array hint).

    Returns
    -------
    ShapeSpec | None
        The accepted shapes, or `None` if any shape is accepted (the
        hint is missing, `Any`, a `TypeVar`, a bare `tuple`, or
        `Tuple[int, ...]`).

    Raises
    ------
    TypeError
        If a `Literal` dimension holds something other than integers, or
        the hint unpacks more than one variadic tuple.
    """
    hint = unwrap(hint)
    if hint is tuple or hint is tx.Tuple:
        return None
    if safe_get_origin(hint) is not tuple:
        return None
    args = safe_get_args(hint)
    if args == ((),):
        # Python 3.8 spells `Tuple[()]` with an inner empty tuple.
        args = ()
    if len(args) == 2 and args[1] is Ellipsis:
        star = _dim_spec(args[0])
        return None if star is None else ShapeSpec(star=star, variadic=True)

    head: tx.List[DimSpec] = []
    tail: tx.List[DimSpec] = []
    star: DimSpec = None
    variadic = False
    for arg in args:
        unpacked = _unpacked(arg)
        if unpacked is None:
            (tail if variadic else head).append(_dim_spec(arg))
            continue
        inner = _unpacked_args(unpacked)
        if inner is None or (len(inner) == 2 and inner[1] is Ellipsis):
            if variadic:
                raise TypeError(
                    f"A shape hint may unpack one variadic tuple only: {hint}"
                )
            variadic = True
            star = None if inner is None else _dim_spec(inner[0])
        else:
            # A fixed-length tuple is spliced in place.
            dims = [_dim_spec(x) for x in inner]
            (tail if variadic else head).extend(dims)
    if variadic and star is None and not head and not tail:
        return None
    return ShapeSpec(head, star, tail, variadic)


def _unpacked(hint: tx.Any) -> tx.Any:
    """
    What `Unpack[X]` or `*X` unpacks, or `None` if `hint` does not
    unpack anything.
    """
    if safe_get_origin(hint) is tx.Unpack:
        return tx.get_args(hint)[0]
    if getattr(hint, "__unpacked__", False):
        # `*tuple[...]` (Python 3.11+) is the tuple itself, flagged.
        return hint
    return None


def _unpacked_args(hint: tx.Any) -> tx.Optional[tx.Tuple[tx.Any, ...]]:
    """
    The arguments of an unpacked tuple hint, or `None` for an unpacked
    `TypeVarTuple` (any number of dimensions of any size).
    """
    if safe_get_origin(hint) is not tuple:
        return None
    args = safe_get_args(hint)
    return () if args == ((),) else args


def _dim_spec(hint: tx.Any) -> DimSpec:
    """The sizes a dimension hint accepts, or `None` for any size."""
    if isinstance(hint, int) and not isinstance(hint, bool):
        return (hint,)
    hint = unwrap(hint)
    if safe_get_origin(hint) is not tx.Literal:
        return None
    sizes = tx.get_args(hint)
    for size in sizes:
        if not isinstance(size, int) or isinstance(size, bool):
            raise TypeError(f"A dimension size must be an integer: {hint}")
    return tuple(sizes)


def _size_ok(size: tx.Any, sizes: DimSpec) -> bool:
    """Whether a dimension of size `size` matches `sizes`."""
    if sizes is None or size in sizes:
        return True
    # Lazy arrays may not know their size yet (dask uses `nan`).
    return size != size


def _dim_repr(sizes: DimSpec) -> str:
    """A short representation of a dimension constraint."""
    if sizes is None:
        return "int"
    return "|".join(map(str, sizes))

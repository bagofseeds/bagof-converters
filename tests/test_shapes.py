# stdlib
import sys

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.converters._shapes import ShapeSpec, shape_spec

L = tx.Literal
Ts = tx.TypeVarTuple("Ts")


# --- shape_spec -------------------------------------------------------


@pytest.mark.parametrize(
    "hint",
    [
        tx.Any,
        tuple,
        tx.Tuple,
        tx.Tuple[int, ...],
        tx.Tuple[tx.Any, ...],
        tx.TypeVar("S"),
        tx.Tuple[tx.Unpack[tx.Tuple[int, ...]]],
        tx.Tuple[tx.Unpack[Ts]],
    ],
)
def test_unconstrained(hint: tx.Any) -> None:
    assert shape_spec(hint) is None


@pytest.mark.parametrize(
    "hint,spec",
    [
        (tx.Tuple[()], ShapeSpec()),
        (tx.Tuple[int], ShapeSpec((None,))),
        (tx.Tuple[int, int], ShapeSpec((None, None))),
        (tx.Tuple[L[3], int], ShapeSpec(((3,), None))),
        (tx.Tuple[L[1, 3]], ShapeSpec(((1, 3),))),
        (tx.Tuple[tx.Annotated[L[3], "x"]], ShapeSpec(((3,),))),
        (tx.Tuple[L[3], ...], ShapeSpec(star=(3,), variadic=True)),
        (
            tx.Tuple[tx.Unpack[tx.Tuple[int, ...]], L[3]],
            ShapeSpec(tail=((3,),), variadic=True),
        ),
        (
            tx.Tuple[L[2], tx.Unpack[Ts], L[3]],
            ShapeSpec(((2,),), None, ((3,),), variadic=True),
        ),
        (
            tx.Tuple[int, tx.Unpack[tx.Tuple[L[4], ...]], int],
            ShapeSpec((None,), (4,), (None,), variadic=True),
        ),
        (
            # A fixed-length unpacked tuple is spliced in place.
            tx.Tuple[int, tx.Unpack[tx.Tuple[L[2], L[3]]]],
            ShapeSpec((None, (2,), (3,))),
        ),
    ],
)
def test_constrained(hint: tx.Any, spec: ShapeSpec) -> None:
    assert shape_spec(hint) == spec


@pytest.mark.skipif(sys.version_info < (3, 11), reason="star syntax")
def test_star_syntax() -> None:
    hint = eval("tuple[int, *tuple[int, ...], tx.Literal[3]]")
    assert shape_spec(hint) == ShapeSpec((None,), None, ((3,),), True)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="builtin generics")
def test_builtin_tuple() -> None:
    assert shape_spec(eval("tuple[tx.Literal[3], int]")) == ShapeSpec(
        ((3,), None)
    )
    assert shape_spec(eval("tuple[()]")) == ShapeSpec()


@pytest.mark.parametrize(
    "hint",
    [
        tx.Tuple[L["a"]],
        tx.Tuple[L[True]],
        tx.Tuple[
            tx.Unpack[tx.Tuple[int, ...]], int, tx.Unpack[tx.Tuple[int, ...]]
        ],
    ],
)
def test_invalid(hint: tx.Any) -> None:
    with pytest.raises(TypeError):
        shape_spec(hint)


# --- ShapeSpec --------------------------------------------------------


def test_eq() -> None:
    assert ShapeSpec(((3,),)) == ShapeSpec(((3,),))
    assert ShapeSpec(((3,),)) != ShapeSpec(((4,),))
    assert ShapeSpec() != ShapeSpec(variadic=True)
    assert ShapeSpec() != ()
    # `star` only matters for a variadic shape
    assert ShapeSpec(star=(3,)) == ShapeSpec()
    assert hash(ShapeSpec(star=(3,))) == hash(ShapeSpec())


def test_repr() -> None:
    assert repr(ShapeSpec()) == "()"
    assert repr(ShapeSpec(((3,),))) == "(3,)"
    assert repr(ShapeSpec((None, (1, 3)))) == "(int, 1|3)"
    assert repr(ShapeSpec(tail=((3,),), variadic=True)) == "(*int, 3)"


@pytest.mark.parametrize(
    "spec,shape,ok",
    [
        (ShapeSpec((None, (3,))), (5, 3), True),
        (ShapeSpec((None, (3,))), (5, 4), False),
        (ShapeSpec((None, (3,))), (3,), False),
        (ShapeSpec((None, (3,))), (1, 5, 3), False),
        (ShapeSpec(((1, 3),)), (1,), True),
        (ShapeSpec(tail=((3,),), variadic=True), (3,), True),
        (ShapeSpec(tail=((3,),), variadic=True), (7, 8, 3), True),
        (ShapeSpec(tail=((3,),), variadic=True), (), False),
        (ShapeSpec(star=(2,), variadic=True), (2, 2, 2), True),
        (ShapeSpec(star=(2,), variadic=True), (2, 3), False),
        # dask may not know a size yet
        (ShapeSpec(((3,),)), (float("nan"),), True),
    ],
)
def test_check(spec: ShapeSpec, shape: tx.Tuple[int, ...], ok: bool) -> None:
    assert (spec.check(shape) is None) is ok


@pytest.mark.parametrize(
    "spec,shape,target",
    [
        # left-pad with singletons
        (ShapeSpec((None, None)), (5,), (1, 5)),
        (ShapeSpec((None, None)), (), (1, 1)),
        # broadcast singletons to a Literal size
        (ShapeSpec(((4,), (4,))), (), (4, 4)),
        (ShapeSpec(((4,), (4,))), (4,), (4, 4)),
        (ShapeSpec(((4,), (4,))), (4, 1), (4, 4)),
        (ShapeSpec((None, (3,))), (5, 1), (5, 3)),
        # a singleton is kept where 1 is allowed
        (ShapeSpec(((1, 3),)), (1,), (1,)),
        # variadic: pad to the smallest number of dimensions only
        (ShapeSpec(tail=((3,),), variadic=True), (), (3,)),
        (ShapeSpec(tail=((3,),), variadic=True), (2, 1), (2, 3)),
        (ShapeSpec(star=(2,), variadic=True), (1, 1), (2, 2)),
    ],
)
def test_broadcast_shape(
    spec: ShapeSpec,
    shape: tx.Tuple[int, ...],
    target: tx.Tuple[int, ...],
) -> None:
    assert spec.broadcast_shape(shape) == target


@pytest.mark.parametrize(
    "spec,shape",
    [
        (ShapeSpec(((4,), (4,))), (3,)),
        (ShapeSpec(((4,), (4,))), (2, 4, 4)),
        # ambiguous: 1 could become 2 or 3
        (ShapeSpec(((2, 3),)), (1,)),
    ],
)
def test_broadcast_shape_invalid(
    spec: ShapeSpec, shape: tx.Tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match=r"\(\d"):
        spec.broadcast_shape(shape)


@pytest.mark.parametrize(
    "spec,shape",
    [
        (ShapeSpec(), ()),
        (ShapeSpec(((3,), None)), (3, 0)),
        (ShapeSpec(((1, 3),)), (1,)),
        (ShapeSpec((None,), (3,), ((2,),), variadic=True), (0, 2)),
    ],
)
def test_default_shape(spec: ShapeSpec, shape: tx.Tuple[int, ...]) -> None:
    assert spec.default_shape() == shape

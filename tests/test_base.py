"""Tests for the base converter machinery."""

# dependencies
import typing_extensions as tx

# locals
from bagof.converters.base import Converter


def test_register_accepts_several_hints() -> None:
    """`register` takes any number of hints, as its docstring says."""
    # The overloads annotated `*hints` as `Unpack[Tuple[Any]]` -- a
    # *one*-element tuple, so "exactly one hint" -- while the
    # implementation and the docstring both say "one or more".
    registry: tx.Dict[tx.Any, tx.Any] = {}

    class _A:
        pass

    class _B:
        pass

    class Multi(Converter):
        DEFAULT = _A

    Converter.register(Multi, _A, _B, registry=registry)
    assert registry[_A] is Multi
    assert registry[_B] is Multi

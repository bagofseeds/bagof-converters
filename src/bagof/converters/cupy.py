"""Converters for cupy array types."""

__all__: list = []

# dependencies
import typing_extensions as tx

# locals
from ._arrays import ArrayConverter
from .base import _has_module, _lazy

if tx.TYPE_CHECKING:
    # Import the bare module so mkdocstrings resolves the `cupy.*`
    # cross-references in the docstrings below. Type-checking only.
    import cupy  # noqa: F401


# cupy is only imported once a cupy converter is actually used: the
# registry keys are forward references, resolved once `cupy` (or the hint
# module) has been imported by the caller -- see `Converter.register`.
if tx.TYPE_CHECKING or _has_module("cupy"):  # pragma: no cover
    # cupy needs a CUDA toolchain and cannot be installed on a CPU runner,
    # so this converter is only ever type-checked, never exercised in CI.

    class ToCupyArray(
        ArrayConverter,
        register=(
            tx.ForwardRef("cupy.ndarray"),
            tx.ForwardRef("bagof.hints.cupy.ndarray"),
        ),
    ):
        """Converter for [`cupy.ndarray`][]."""

        DEFAULT = _lazy("cupy.ndarray")
        ARRAY = SCALARS = _lazy("cupy")
        ARRAY_TYPE = _lazy("cupy.ndarray")
        HINT_TYPE = _lazy("bagof.hints.cupy.ndarray")

    __all__ += ["ToCupyArray"]

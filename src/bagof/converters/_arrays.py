"""
Shared base for the array converters (numpy / cupy / dask).

The three array libraries convert almost identically: coerce the value to
the library's array type, coerce its dtype to the one named in the hint,
then reinterpret it as the requested subclass. This module factors that out
so the per-library modules stay a few lines each.
"""

__all__ = ["ArrayConverter", "scalar_maps"]

# stdlib
import numbers

# dependencies
import typing_extensions as tx

# bags
from bagof.core.magic import get_args_uw, safe_isinstance, safe_issubclass

# locals
from .base import Converter


def scalar_maps(xp: tx.Any) -> tx.Tuple[dict, dict]:
    """
    Build the dtype lookup tables for an array/scalar namespace ``xp``.

    Returns
    -------
    py_to_scalar : dict
        Maps Python (and [`numbers`][]) types to the namespace's scalar
        type, e.g. ``float -> xp.floating``.
    abstract_to_py : dict
        Maps abstract scalar types (which cannot be passed to ``astype``)
        to a concrete Python type, e.g. ``xp.floating -> float``.
    """
    py_to_scalar = {
        object: xp.object_,
        str: xp.str_,
        bool: xp.bool_,
        int: xp.integer,
        float: xp.floating,
        complex: xp.complexfloating,
        numbers.Number: xp.number,
        numbers.Complex: xp.complexfloating,
        numbers.Real: xp.floating,
        numbers.Rational: xp.floating,
        numbers.Integral: xp.integer,
    }
    abstract_to_py = {
        xp.generic: None,
        xp.number: None,
        xp.complexfloating: complex,
        xp.floating: float,
        xp.integer: int,
    }
    return py_to_scalar, abstract_to_py


class ArrayConverter(Converter[tx.Any, tx.Any]):
    """
    Base for array converters.

    A subclass sets the class attributes below; it does not override
    ``__call__``. The array/scalar namespaces are held as callables so the
    class body never touches an optional library at import time.

    !!! note
        The concrete converters (``ToNDArray``, ``ToDaskArray``,
        ``ToCupyArray``) are only defined and registered when their
        backing library (numpy, dask or cupy) can be imported.
    """

    #: The array namespace (numpy / cupy / dask.array) -- provides
    #: ``asanyarray`` and the concrete array type.
    ARRAY: tx.Any = None
    #: The scalar-type namespace (numpy, or cupy for cupy arrays) used to
    #: build the dtype tables. dask reuses numpy's, since dask arrays hold
    #: numpy dtypes.
    SCALARS: tx.Any = None
    #: The concrete array type (``numpy.ndarray`` / ``cupy.ndarray`` /
    #: ``dask.array.Array``).
    ARRAY_TYPE: tx.Any = None
    #: The ``bagof.hints`` stub for the array type. On Python 3.8 (where the
    #: real type is not subscriptable) a parametrised hint carries the stub
    #: as its origin, so it is mapped back to ``ARRAY_TYPE``.
    HINT_TYPE: tx.Any = None
    #: Whether the library supports ``array.view(cls)`` to reinterpret as a
    #: subclass. dask does not, and uses the subclass constructor instead.
    CAN_VIEW: bool = True

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """Accept anything array-like (implements ``__array__``)."""
        from bagof.hints.array import ArrayLike

        return ArrayLike

    @property
    def _array_origin(self) -> tx.Any:
        # Map the hint stub back to the real array type (see HINT_TYPE).
        origin = self.origin
        if origin is self.HINT_TYPE:
            return self.ARRAY_TYPE
        return origin

    @property
    def _target_scalar(self) -> tx.Any:
        # The scalar type named by ``ndarray[shape, dtype[X]]`` -> ``X``.
        args = self.args
        if len(args) < 2:
            return None
        dtype_args = get_args_uw(args[1])
        if not dtype_args:
            return None
        py_to_scalar, _ = scalar_maps(self.SCALARS)
        scalar = dtype_args[0]
        return py_to_scalar.get(scalar, scalar)

    def __call__(self, value: tx.Any) -> tx.Any:
        """Convert the value to the requested array type and dtype."""
        xp, array_type = self.ARRAY, self.ARRAY_TYPE
        origin = self._array_origin
        scalar = self._target_scalar
        _, abstract_to_py = scalar_maps(self.SCALARS)

        convert = self._wrap_converter(
            lambda v: self._to_array(v, xp, array_type, origin, scalar,
                                     abstract_to_py)
        )
        return convert(value)

    def _to_array(
        self,
        value: tx.Any,
        xp: tx.Any,
        array_type: tx.Any,
        origin: tx.Any,
        scalar: tx.Any,
        abstract_to_py: dict,
    ) -> tx.Any:
        # 1. Ensure it is one of the library's arrays (using the target
        #    dtype up front when we have a concrete one).
        if not safe_isinstance(value, array_type):
            value = xp.asanyarray(value, abstract_to_py.get(scalar, scalar))
        # 2. Coerce the dtype if it does not already match.
        if scalar is not None and not safe_issubclass(value.dtype.type,
                                                      scalar):
            value = value.astype(abstract_to_py.get(scalar, scalar))
        # 3. Reinterpret as the requested subclass if needed.
        if not safe_isinstance(value, origin):
            value = value.view(origin) if self.CAN_VIEW else origin(value)
        return value

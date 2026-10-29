"""
Shared base for the array converters (numpy / cupy / dask).

The three array libraries convert almost identically: move the value to
where the library can read it, coerce it to the library's array type,
coerce its dtype to the one named in the hint, broadcast it to the shape
named in the hint, then reinterpret it as the requested subclass. This
module factors that out so the per-library modules stay a few lines each.
"""

__all__ = ["ArrayConverter", "scalar_maps", "compute", "to_host"]

# stdlib
import numbers

# dependencies
import typing_extensions as tx

# bags
from bagof.core.magic import get_args_uw, safe_isinstance, safe_issubclass

# locals
from ._shapes import ShapeSpec, shape_spec
from .base import Converter

#: The methods that copy a device (GPU) array to the host, by library:
#: cupy (``get``), numba (``copy_to_host``) and torch (``cpu``).
_TO_HOST = ("get", "copy_to_host", "cpu")


def compute(value: tx.Any) -> tx.Any:
    """
    Compute a lazy (dask) array, or return any other value unchanged.

    Lazy arrays are recognised by the dask collection protocol
    (``__dask_graph__``), so that dask itself is never imported.
    """
    if callable(getattr(value, "__dask_graph__", None)) and callable(
        getattr(value, "compute", None)
    ):
        return value.compute()
    return value


def to_host(value: tx.Any) -> tx.Any:
    """
    Compute a lazy array, then copy it to the host if it lives on a
    device; return any other value unchanged.

    Device arrays are recognised by the CUDA array interface
    (``__cuda_array_interface__``), so that cupy itself is never
    imported. They refuse an implicit conversion to numpy, so they are
    copied explicitly with their library's own method (see
    ``_TO_HOST``).
    """
    value = compute(value)
    # Look the interface up on the class: reading it on the instance runs
    # a property that may raise (torch does, for a tensor needing grad).
    if getattr(type(value), "__cuda_array_interface__", None) is not None:
        for name in _TO_HOST:
            method = getattr(value, name, None)
            if callable(method):
                return method()
    return value


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
    ``__call__``. An optional library's namespaces may be given lazily
    (see [`lazy_import`][bagof.core.magic.lazy_import]) so the class body
    never imports it.

    !!! note
        The concrete converters (``ToNDArray``, ``ToDaskArray``,
        ``ToCupyArray``) are only defined and registered when their
        backing library (numpy, dask or cupy) is installed.
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

    def move(self, value: tx.Any) -> tx.Any:
        """
        Move a value that is not one of the library's arrays to where
        ``ARRAY.asanyarray`` can read it (e.g. from the GPU to the host).

        The default leaves it alone; a subclass overrides it with
        `to_host` or `compute` (from this module).
        """
        return value

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
        if scalar is tx.Any or not isinstance(scalar, type):
            # `Any` or a `TypeVar` (as in the bare `NDArray` alias).
            return None
        return py_to_scalar.get(scalar, scalar)

    @property
    def _target_shape(self) -> tx.Optional[ShapeSpec]:
        # The shapes named by ``ndarray[shape, dtype]``, or None for any.
        args = self.args
        if not args:
            return None
        return shape_spec(args[0])

    def __call__(self, value: tx.Any) -> tx.Any:
        """
        Convert the value to the requested array type, dtype and shape.

        An array of another library is moved first (see `move`). An array
        with fewer dimensions than the shape hint names is left-padded
        with singleton dimensions, and a singleton dimension is broadcast
        to the size a `Literal` names, following numpy's broadcasting
        rules.

        !!! note
            Broadcasting returns a view that shares its memory across
            the broadcast dimensions (read-only, for numpy), rather than
            a copy.
        """
        xp, array_type = self.ARRAY, self.ARRAY_TYPE
        origin = self._array_origin
        scalar = self._target_scalar
        shape = self._target_shape
        _, abstract_to_py = scalar_maps(self.SCALARS)

        convert = self._wrap_converter(
            lambda v: self._to_array(v, xp, array_type, origin, scalar,
                                     abstract_to_py, shape)
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
        shape: tx.Optional[ShapeSpec] = None,
    ) -> tx.Any:
        # 1. Ensure it is one of the library's arrays (using the target
        #    dtype up front when we have a concrete one), moving it from
        #    another library first.
        if not safe_isinstance(value, array_type):
            value = self.move(value)
            value = xp.asanyarray(value, abstract_to_py.get(scalar, scalar))
        # 2. Coerce the dtype if it does not already match.
        if scalar is not None and not safe_issubclass(value.dtype.type,
                                                      scalar):
            value = value.astype(abstract_to_py.get(scalar, scalar))
        # 3. Pad and broadcast to the requested shape if needed.
        if shape is not None:
            value = self._to_shape(value, xp, shape)
        # 4. Reinterpret as the requested subclass if needed.
        if not safe_isinstance(value, origin):
            value = value.view(origin) if self.CAN_VIEW else origin(value)
        return value

    def _to_shape(self, value: tx.Any, xp: tx.Any, shape: ShapeSpec) -> tx.Any:
        # Left-pad with (or strip) singleton dimensions, then broadcast.
        try:
            target = shape.broadcast_shape(value.shape)
        except ValueError as e:
            raise self.value_error(value, str(e)) from None
        if len(target) > value.ndim:
            value = value[(None,) * (len(target) - value.ndim)]
        elif len(target) < value.ndim:
            value = value.reshape(value.shape[value.ndim - len(target):])
        if tuple(target) != tuple(value.shape):
            value = xp.broadcast_to(value, target)
        return value

# dependencies
import typing_extensions as tx

# optional
if tx.TYPE_CHECKING:
    from types import NoneType, UnionType

else:
    try:
        from types import NoneType
    except ImportError:  # pragma: no cover
        NoneType = type(None)  # type: ignore[assignment]

    try:
        from types import UnionType
    except ImportError:  # pragma: no cover
        UnionType = tx.Union  # type: ignore[assignment]

if tx.TYPE_CHECKING:
    import numpy as np

else:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None  # type: ignore[assignment]

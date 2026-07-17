"""
Automatic type-based converters.

Modules
-------
base
    Base class for magic converters.
collections
    Converters for collection types (list, tuple, dict, etc.).
common
    Common converters (any, union, etc.).
enums
    Converters for enumeration types.
exceptions
    Exceptions raised by converters on conversion error.
numbers
    Converters for numeric types (int, float, etc.).
numpy
    Converters for numpy types (dtype, etc.).
pandas
    Converters for pandas types (DataFrame, Series).
strings
    Converters for string types (str, bytes, etc.).
"""

__all__ = [
    "__version__",
    "base",
    "collections",
    "common",
    "enums",
    "exceptions",
    "numbers",
    "numpy",
    "pandas",
    "strings",
]

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0+unknown"

from . import (
    base,
    collections,
    common,
    enums,
    exceptions,
    numbers,
    numpy,
    pandas,
    strings,
)
from .base import *  # noqa: F401, F403
from .base import __all__ as __all_base
from .collections import *  # noqa: F401, F403
from .collections import __all__ as __all_collections
from .common import *  # noqa: F401, F403
from .common import __all__ as __all_common
from .enums import *  # noqa: F401, F403
from .enums import __all__ as __all_enums
from .exceptions import *  # noqa: F401, F403
from .exceptions import __all__ as __all_exceptions
from .numbers import *  # noqa: F401, F403
from .numbers import __all__ as __all_numbers
from .numpy import *  # noqa: F401, F403
from .numpy import __all__ as __all_numpy
from .pandas import *  # noqa: F401, F403
from .pandas import __all__ as __all_pandas
from .strings import *  # noqa: F401, F403
from .strings import __all__ as __all_strings

__all__ += __all_base
__all__ += __all_collections
__all__ += __all_common
__all__ += __all_enums
__all__ += __all_exceptions
__all__ += __all_numbers
__all__ += __all_numpy
__all__ += __all_pandas
__all__ += __all_strings

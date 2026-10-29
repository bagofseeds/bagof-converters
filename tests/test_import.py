"""Importing the package must not import optional backends (#55)."""

# stdlib
import subprocess
import sys
import textwrap

# dependencies
import pytest

BACKENDS = ("numpy", "dask", "dask.array", "cupy", "pandas")


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize("package", ["bagof.converters", "bagof.magic"])
def test_import_does_not_import_backends(package: str) -> None:
    out = _run(f"""
        import importlib.util, sys
        try:
            spec = importlib.util.find_spec({package!r})
        except ImportError:
            spec = None
        if spec is None:
            print("SKIP")
            raise SystemExit
        import {package}
        print(sorted(m for m in {BACKENDS!r} if m in sys.modules))
    """)
    if out.strip() == "SKIP":
        pytest.skip(f"{package} is not installed")
    assert out.strip() == "[]"


def test_all_lists_installed_backend_converters() -> None:
    import importlib.util

    import bagof.converters

    expected = {
        "numpy": ["ToDType", "ToNDArray"],
        "dask": ["ToDaskArray"],
        "cupy": ["ToCupyArray"],
        "pandas": ["ToDataFrame", "ToSeries"],
    }
    for module, names in expected.items():
        if importlib.util.find_spec(module) is None:
            continue
        for name in names:
            assert name in bagof.converters.__all__
            assert hasattr(bagof.converters, name)


def test_numpy_converters_resolve_after_late_import() -> None:
    pytest.importorskip("numpy")
    out = _run("""
        from bagof.converters import Converter, ToDType, ToNDArray
        import numpy as np
        assert Converter.get_class(np.dtype) is ToDType
        assert Converter.get_class(np.float32) is ToDType
        assert Converter.get_class(np.ndarray) is ToNDArray
        print(Converter.get(np.ndarray)([1, 2]).tolist(),
              Converter.get(np.dtype)("float32"))
    """)
    assert out.strip() == "[1, 2] float32"


def test_numpy_hint_resolves_after_late_import() -> None:
    pytest.importorskip("numpy")
    out = _run("""
        from bagof.converters import Converter
        from bagof.hints.numpy import NDArray
        x = Converter.get(NDArray[float])([1, 2])
        print(type(x).__module__, x.dtype)
    """)
    assert out.strip() == "numpy float64"


def test_numbers_accept_numpy_scalars_after_late_import() -> None:
    pytest.importorskip("numpy")
    out = _run("""
        import numbers
        import typing
        from bagof.converters import Converter
        import numpy as np
        expected = typing.Union[numbers.Integral, np.integer, np.bool_]
        print(Converter.get(int).like() == expected)
    """)
    assert out.strip() == "True"


def test_dask_converters_resolve_after_late_import() -> None:
    pytest.importorskip("dask.array")
    out = _run("""
        from bagof.converters import Converter, ToDaskArray
        import dask.array as da
        assert Converter.get_class(da.Array) is ToDaskArray
        x = Converter.get(da.Array)([1, 2, 3])
        assert isinstance(x, da.Array)
        print(x.compute().tolist())
    """)
    assert out.strip() == "[1, 2, 3]"


def test_dask_hint_resolves_after_late_import() -> None:
    pytest.importorskip("dask.array")
    out = _run("""
        from bagof.converters import Converter
        from bagof.hints.dask import NDArray
        x = Converter.get(NDArray[float])([1, 2])
        print(type(x).__module__.split(".")[0], x.dtype)
    """)
    assert out.strip() == "dask float64"


def test_pandas_converters_resolve_after_late_import() -> None:
    pytest.importorskip("pandas")
    out = _run("""
        from bagof.converters import Converter, ToDataFrame, ToSeries
        import pandas as pd
        assert Converter.get_class(pd.DataFrame) is ToDataFrame
        assert Converter.get_class(pd.Series) is ToSeries
        print(Converter.get(pd.Series)([1, 2]).tolist())
    """)
    assert out.strip() == "[1, 2]"

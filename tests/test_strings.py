# stdlib
import re

# dependencies
import pytest
import typing_extensions as tx

# locals
from bagof.converters import strings
from bagof.converters.exceptions import ConversionError

# --- ToString ---------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("hello", "hello"),
        ("", ""),
    ],
)
def test_to_str_valid(value: tx.Any, expected: str) -> None:
    converter = strings.ToString()
    assert converter(value) == expected


@pytest.mark.parametrize(
    "value",
    [1, None, [1, 2], {"a": 1}],
)
def test_to_str_invalid(value: tx.Any) -> None:
    converter = strings.ToString()
    with pytest.raises(ConversionError):
        converter(value)


# --- ToRegexMatch -----------------------------------------------------


@pytest.mark.parametrize(
    "pattern,value",
    [
        (r"\d+", "123"),
        (r"[a-z]+", "abc"),
        (re.compile(r"\w+"), "hello_world"),
    ],
)
def test_regex_match_valid(
    pattern: tx.Union[str, re.Pattern], value: str
) -> None:
    converter = strings.ToRegexMatch(pattern)
    assert converter(value) == value


@pytest.mark.parametrize(
    "pattern,value",
    [
        (r"\d+", "abc"),
        (r"[a-z]+", "ABC"),
        (r"\d+", ""),
    ],
)
def test_regex_match_invalid(
    pattern: tx.Union[str, re.Pattern], value: str
) -> None:
    converter = strings.ToRegexMatch(pattern)
    with pytest.raises(ConversionError):
        converter(value)


def test_annotated_regex() -> None:
    """ToRegexMatch should work as Annotated metadata."""
    from bagof.converters.common import ToAnnotated

    hint = tx.Annotated[str, re.compile(r"\d+")]
    converter = ToAnnotated(hint)
    assert converter("42") == "42"
    with pytest.raises(ConversionError):
        converter("abc")

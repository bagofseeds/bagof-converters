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
        # Bytes are decoded (`like` advertises `Union[str, bytes]`).
        (b"hello", "hello"),
        (b"", ""),
        ("héllo".encode(), "héllo"),
    ],
)
def test_to_str_valid(value: tx.Any, expected: str) -> None:
    converter = strings.ToString()
    assert converter(value) == expected


@pytest.mark.parametrize(
    "value",
    [1, None, [1, 2], {"a": 1}, bytearray(b"ab")],
)
def test_to_str_invalid(value: tx.Any) -> None:
    converter = strings.ToString()
    with pytest.raises(ConversionError):
        converter(value)


def test_to_str_decodes_bytes() -> None:
    # Regression: the guard read `not isinstance(v, str) or isinstance(v,
    # bytes)`, which raised for *every* bytes input and left the decoding
    # branch below it unreachable.
    result = strings.ToString()(b"hello")
    assert result == "hello"
    assert type(result) is str


def test_to_str_accepts_what_like_advertises() -> None:
    # `like` promises str and bytes; `__call__` must honour both.
    assert strings.ToString().like() == tx.Union[str, bytes]
    strings.ToString()("hello")
    strings.ToString()(b"hello")


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
    "pattern,value,expected",
    [
        (r"\d+", b"123", "123"),
        (r"[a-z]+", b"abc", "abc"),
    ],
)
def test_regex_match_decodes_bytes(
    pattern: tx.Union[str, re.Pattern], value: bytes, expected: str
) -> None:
    # `ToRegexMatch` matches against the decoded string, inherited from
    # `ToString`.
    converter = strings.ToRegexMatch(pattern)
    assert converter(value) == expected


def test_regex_match_invalid_bytes() -> None:
    converter = strings.ToRegexMatch(r"\d+")
    with pytest.raises(ConversionError):
        converter(b"abc")


def test_annotated_regex_decodes_bytes() -> None:
    from bagof.converters.common import ToAnnotated

    hint = tx.Annotated[str, re.compile(r"\d+")]
    assert ToAnnotated(hint)(b"42") == "42"


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

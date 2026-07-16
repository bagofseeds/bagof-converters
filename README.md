# bagof-converters

Hint-based converters that operate at runtime.

## Example

```pycon
>>> from bagof.converters import get_converter
>>> convert = get_converter(list[int])
>>> convert(range(3))
[0, 1, 2]
>>> convert(1)
ValueConversionError: ToSequence(list[int]): Invalid value.
|> value = 1
>>> convert(["a", "b", "c"])
ValueConversionError: ToNumber(<class 'int'>): Invalid value.
|> value = 'a'
The above exception was the direct cause of the following exception:
ValueConversionError: ToSequence(list[int]): Invalid value.
|> value = 'a'
The above exception was the direct cause of the following exception:
ValueConversionError: ToSequence(list[int]): Invalid value.
|> value = <map object at 0x72fb38570cd0>
```

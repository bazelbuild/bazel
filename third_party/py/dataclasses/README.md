# dataclasses

This is the dataclasses.py module from the Python 3.7 standard library,
backported to Python 3.6.

Example usage:

```python
import dataclasses

@dataclasses.dataclass
class Foo:
  foo: str  # The Foo() constructor will require a value for this field.
  bar: int = 42  # Default value if one is not passed to the constructor.

  def sum_one(self) -> int:
    return self.bar + 1
```

More details: https://www.python.org/dev/peps/pep-0557/

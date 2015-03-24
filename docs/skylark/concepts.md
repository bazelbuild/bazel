Concepts
========

Skylark is the code name of the extension mechanism.
Skylark is a superset of the core build language and its syntax is a subset of
Python. The following constructs have been added to the core build language:
`if` statements, `for` loops, and function definitions.
It is designed to be simple, thread-safe and integrated with the
BUILD language. It is not a general-purpose language and most Python
features are not included.


Some differences with Python should be noted:

* All data structures are immutable.

* All global values are constant (they cannot be reassigned).

* Heterogeneous lists and dictionaries are forbidden.

* The type of a variable may not change, e.g. this is forbidden:
  `a = 2; a = "str"`

* `x += y` is syntactic sugar for `x = x + y`. Even if `x` and `y` are lists,
  dicts or sets, the original value is not mutated, so references to `x`
  that were assigned before the operation will see the old value.

* The + operator is defined for dictionaries, returning an immutable
  concatenated dictionary created from the entries of the original
  dictionaries. In case of duplicate keys, we use values from the second
  operand.

* Dictionary assignment has slightly different semantics: `d["x"] = y` is
  syntactic sugar for `d = d + {"x": y}` or `d += {"x": y}`.

* Recursion is not allowed.

The following Python features are not supported:

* `class` (see `struct` function)
* `import` (see `load` statement)
* `while`, `yield`, `break`, `continue`
* `lambda`
* `try`, `raise`, `except`, `finally` (see `fail` for fatal errors).
* most builtin functions, most methods



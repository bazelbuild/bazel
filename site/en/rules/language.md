Project: /_project.yaml
Book: /_book.yaml

# Starlark Language

{% include "_buttons.html" %}

<!-- [TOC] -->

This page is an overview of [Starlark](https://github.com/bazelbuild/starlark),
formerly known as Skylark, the language used in Bazel. For a complete list of
functions and types, see the [Bazel API reference](/rules/lib/overview).

For more information about the language, see [Starlark's GitHub repo](https://github.com/bazelbuild/starlark/).

For the authoritative specification of the Starlark syntax and
behavior, see the [Starlark Language Specification](https://github.com/bazelbuild/starlark/blob/master/spec.md).

## Syntax

Starlark's syntax is inspired by Python3. This is valid syntax in Starlark:

```python
def fizz_buzz(n):
  """Print Fizz Buzz numbers from 1 to n."""
  for i in range(1, n + 1):
    s = ""
    if i % 3 == 0:
      s += "Fizz"
    if i % 5 == 0:
      s += "Buzz"
    print(s if s else i)

fizz_buzz(20)
```

Starlark's semantics can differ from Python, but behavioral differences are
rare, except for cases where Starlark raises an error. The following Python
types are supported:

* [None](lib/globals#None)
* [bool](lib/bool)
* [dict](lib/dict)
* [tuple](lib/tuple)
* [function](lib/function)
* [int](lib/int)
* [list](lib/list)
* [string](lib/string)

## Mutability

Starlark favors immutability. Two mutable data structures are available:
[lists](lib/list) and [dicts](lib/dict). Changes to mutable
data-structures, such as appending a value to a list or deleting an entry in a
dictionary are valid only for objects created in the current context. After a
context finishes, its values become immutable.

This is because Bazel builds use parallel execution. During a build, each `.bzl`
file and each `BUILD` file get their own execution context. Each rule is also
analyzed in its own context.

Let's go through an example with the file `foo.bzl`:

```python
# `foo.bzl`
var = [] # declare a list

def fct(): # declare a function
  var.append(5) # append a value to the list

fct() # execute the fct function
```

Bazel creates `var` when `foo.bzl` loads. `var` is thus part of `foo.bzl`'s
context. When `fct()` runs, it does so within the context of `foo.bzl`. After
evaluation for `foo.bzl` completes, the environment contains an immutable entry,
`var`, with the value `[5]`.

When another `bar.bzl` loads symbols from `foo.bzl`, loaded values remain
immutable. For this reason, the following code in `bar.bzl` is illegal:

```python
# `bar.bzl`
load(":foo.bzl", "var", "fct") # loads `var`, and `fct` from `./foo.bzl`

var.append(6)  # runtime error, the list stored in var is frozen

fct()          # runtime error, fct() attempts to modify a frozen list
```

Global variables defined in `bzl` files cannot be changed outside of the
`bzl` file that defined them. Just like the above example using `bzl` files,
values returned by rules are immutable.

## Differences between BUILD and .bzl files {:#differences-between-build-and-bzl-files}

`BUILD` files register targets via making calls to rules. `.bzl` files provide
definitions for constants, rules, macros, and functions.

[Native functions](/reference/be/functions) and [native rules](
/reference/be/overview#language-specific-native-rules) are global symbols in
`BUILD` files. `bzl` files need to load them using the [`native` module](
/rules/lib/toplevel/native).

There are two syntactic restrictions in `BUILD` files: 1) declaring functions is
illegal, and 2) `*args` and `**kwargs` arguments are not allowed.

## Differences with Python

* Global variables are immutable.

* `for` statements are not allowed at the top-level. Use them within functions
  instead. In `BUILD` files, you may use list comprehensions.

* `if` statements are not allowed at the top-level. However, `if` expressions
  can be used: `first = data[0] if len(data) > 0 else None`.

* Deterministic order for iterating through Dictionaries.

* Recursion is not allowed.

* Int type is limited to 32-bit signed integers. Overflows will throw an error.

* Modifying a collection during iteration is an error.

* Except for equality tests, comparison operators `<`, `<=`, `>=`, `>`, etc. are
not defined across value types. In short: `5 < 'foo'` will throw an error and
`5 == "5"` will return false.

* In tuples, a trailing comma is valid only when the tuple is between
  parentheses — when you write `(1,)` instead of `1,`.

* Dictionary literals cannot have duplicated keys. For example, this is an
  error: `{"a": 4, "b": 7, "a": 1}`.

* Strings are represented with double-quotes (such as when you call
  [repr](lib/globals#repr)).

* Strings aren't iterable.

The following Python features are not supported:

* implicit string concatenation (use explicit `+` operator).
* Chained comparisons (such as `1 < x < 5`).
* `class` (see [`struct`](lib/struct#struct) function).
* `import` (see [`load`](/extending/concepts#loading-an-extension) statement).
* `while`, `yield`.
* float and set types.
* generators and generator expressions.
* `is` (use `==` instead).
* `try`, `raise`, `except`, `finally` (see [`fail`](lib/globals#fail) for fatal errors).
* `global`, `nonlocal`.
* most builtin functions, most methods.

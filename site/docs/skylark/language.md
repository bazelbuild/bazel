---
layout: documentation
title: Starlark Language
---

# Starlark Language

<!-- [TOC] -->

The page is an overview of [Starlark](https://github.com/bazelbuild/starlark)
(formerly known as Skylark), the language used in Bazel. This should be
enough to get you started. Please refer to the
[Starlark repository on GitHub](https://github.com/bazelbuild/starlark/)
for more information about the language.
For a complete list of functions and types, please check the
[API reference](lib/skylark-overview.html).

## Syntax

Starlark is designed to be small, simple, and thread-safe. Although it is
inspired from Python, it is not a general-purpose language and most Python
features are not included.

Here is an example to show the syntax:

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

The following basic types are supported: [None](lib/globals.html#None),
[bool](lib/bool.html), [dict](lib/dict.html), function, [int](lib/int.html),
[list](lib/list.html), [string](lib/string.html). On top of that, two new
types are specific to Bazel: [depset](lib/depset.html) and
[struct](lib/struct.html).

Starlark is syntactically a subset of Python 3, and will remain so through at
least the 1.x release lifecycle. This ensures that Python-based tooling can at
least parse Starlark code. Although Starlark is not *semantically* a subset of
Python, behavioral differences are rare (excluding cases where Starlark raises
an error).


## Mutability

Because evaluation of BUILD and .bzl files is performed in parallel, there are
some restrictions in order to guarantee thread-safety and determinism. Two
mutable data structures are available: [lists](lib/list.html) and
[dicts](lib/dict.html).

In a build, there are many "evaluation contexts": each `.bzl` file and each
`BUILD` file is loaded in a different context. Each rule is also analyzed in a
separate context. We allow side-effects (e.g. appending a value to a list or
deleting an entry in a dictionary) only on objects created during the current
evaluation context. Once the code in that context is done executing, all of its
values are frozen.

For example, here is the content of the file `foo.bzl`:

```python
var = []

def fct():
  var.append(5)

fct()
```

The variable `var` is created when `foo.bzl` is loaded. `fct()` is called during
the same context, so it is safe. At the end of the evaluation, the environment
contains an entry mapping the identifier `var` to a list `[5]`; this list is
then frozen.

It is possible for multiple other files to load symbols from `foo.bzl` at the
same time. For this reason, the following code is not legal:

```python
load(":foo.bzl", "var", "fct")

var.append(6)  # runtime error, the list stored in var is frozen

fct()          # runtime error, fct() attempts to modify a frozen list
```

Evaluation contexts are also created for the analysis of each custom rule. This
means that any values that are returned from the rule's analysis are frozen.
Note that by the time a custom rule's analysis begins, the .bzl file in which
it is defined has already been loaded, and so the global variables are already
frozen.


## Differences between BUILD and .bzl files

The goal of a BUILD file is to register targets, by making calls to rules. On
the other hand, loading a .bzl file doesn't have any side effect: .bzl files are
only providing definitions (constants or functions).

[Some functions](../be/functions.html) and [native rules](
../be/overview.html#language-specific-native-rules) are exposed to BUILD files
as global symbols. In bzl files, they are available under the [`native` module](
https://docs.bazel.build/versions/master/skylark/lib/native.html).

There are two syntactic restrictions in BUILD files:

* Function definitions (`def` statements) are not allowed in BUILD files.

* `*args` and `**kwargs` arguments are not allowed in BUILD files; instead list
  all the arguments explicitly.


## Differences with Python

In addition to the mutability restrictions, there are also differences with
Python:

* Global variables cannot be reassigned.

* `for` statements are not allowed at the top-level; factor them into functions
  instead. In BUILD files, you may use list comprehensions.

* `if` statements are not allowed at the top-level. However, `if` expressions
  can be used: `first = data[0] if len(data) > 0 else None`.

* Dictionaries have a deterministic order of iteration.

* Recursion is not allowed.

* Int type is limited to 32-bit signed integers (an overflow will throw an
  error).

* Modifying a collection during iteration is an error. You can avoid the error
  by iterating over a copy of the collection, e.g.
  `for x in list(my_list): ...`. You can still modify its deep contents
  regardless.

* The comparison operators (`<`, `<=`, `>=`, `>`) are not defined across
  different types of values, e.g., you can't compare `5 < 'foo'` (however you
  still can compare them using `==` or `!=`). This is a difference with Python
  2, but consistent with Python 3. Note that this means you are unable to sort
  lists that contain mixed types of values.

* Tuple syntax is more restrictive. You may use a trailing comma only when the
  tuple is between parentheses, e.g. write `(1,)` instead of `1,`.

* Dictionary literals cannot have duplicated keys. For example, this is an
  error: `{"a": 4, "b": 7, "a": 1}`.

* Variable of a comprehension may not be used after the comprehension. This is
  stricter than Python 2 and Python 3, which have different behavior (shadowing
  vs reassignment).

* Strings are represented with double-quotes (e.g. when you
  call [repr](lib/globals.html#repr)).

The following Python features are not supported:

*   implicit string concatenation (use explicit `+` operator)
*   Chained comparisons (e.g. `1 < x < 5`)
*   `class` (see [`struct`](lib/struct.html#struct) function)
*   `import` (see [`load`](concepts.md#loading-an-extension) statement)
*   `while`, `yield`
*   float and set types
*   generators and generator expressions
*   `lambda` and nested functions
*   `is` (use `==` instead)
*   `try`, `raise`, `except`, `finally` (see [`fail`](lib/globals.html#fail) for
    fatal errors)
*   `global`, `nonlocal`
*   most builtin functions, most methods

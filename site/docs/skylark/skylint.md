---
layout: documentation
title: Skylint
---

# Skylint

[Style guide](../skylark/bzl-style.md)

This document explains how to use Skylint, the Starlark linter.

<!-- [TOC] -->

## The linter CLI tool

### Building the linter

To build the linter from source, use the following commands:

```
$ git clone https://github.com/bazelbuild/bazel.git
$ cd bazel
$ bazel build //src/tools/skylark/java/com/google/devtools/skylark/skylint:Skylint
```

After that, the linter is available at `bazel-bin/src/tools/skylark/java/com/google/devtools/skylark/skylint/Skylint`.
You can copy it to somewhere else if you prefer.

### Running the linter

Assuming you have the linter binary available at `/path/to/Skylint` from the
previous step, you can run it like this:

```
$ /path/to/Skylint /path/to/bzl/file.bzl
```

where `/path/to/Skylint` is the path of the binary from the previous section.

## The checks in detail

This section explains which checks the linter performs and how to deal with
false positives.

### Deprecating functions (docstring format) [deprecated-symbol]

<a name="deprecated-symbol"></a>
To deprecate a function, add a `Deprecated:` section to the docstring, similarly
to a `Returns:` section.

```
def foo():
  """An example function.

  Deprecated:
    <reason and alternative>
  """
  # …

def bar():
  foo() # warning: "usage of 'foo' is deprecated: <reason and alternative>"
```

Note that the explanation starts on the next line after `Deprecated:` and may
occupy multiple lines, with all lines indented by two spaces.

### Using the operator + on dictionaries [deprecated-plus-dict]

<a name="deprecated-plus-dict"></a>
The `+` operator (and similarly `+=`) is deprecated for dictionaries. Instead,
use the following:

You can import a helper function from [Skylib](https://github.com/bazelbuild/bazel-skylib)
and use it like this:

```
load("@bazel_skylib//lib:dicts.bzl", "dicts")
dicts.add(d1, d2, d3) # instead of d1 + d2 + d3
```

### Using the operator + on depset [deprecated-plus-depset]

<a name="deprecated-plus-depset"></a>
The `+` operator (and similarly `+=`) is deprecated for depsets. Instead,
use the depset constructor.

See [documentation on depsets](depsets.md) for background and examples of use.

```
  d1 = depset(items1)
  d2 = depset(items2)
  combined = depset(transitive=[d1, d2])
```


### Docstrings

<a name="missing-module-docstring"></a>
<a name="missing-function-docstring"></a>
<a name="bad-docstring-format"></a>
Categories: [missing-module-docstring] [missing-function-docstring] [bad-docstring-format]

The Starlark conventions for docstrings are similar to the [the Python
conventions](https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments).
Docstrings are triple-quoted string literals and the closing quotes have to
appear on their own line unless the docstring is only one line long.

A file-level docstring is the first statement of a file, even before the
`load()` statements. A function docstring is the first statement in the function
body.

Sections in the docstring (such as `Args:` and `Returns:`) are separated by a
blank line. Their contents are indented by two spaces.
Example:

```
"""This module contains some docstrings examples."""

def example_function1():
  """Illustrates the usage of a one-line function docstring."""

def example_function2(foo, bar):
  """Illustrates the format of a longer function docstring.

  After the one-line summary comes the description body.
  It contains additional information about the function.

  The description can span more than one paragraph.

  Args:
    foo: documentation of the first parameter.
      Subsequent lines have to be indented by two spaces.
    bar: documentation of the second parameter

  Returns:
    documentation of the return value

  Deprecated:
    This function is deprecated for <reason>. Use <alternative> instead.
  """
  return "baz"
```

#### Indentation

If the linter gives you confusing error messages, **check the indentation of the
docstring**. (The indentation rules are the same as [for
Python](https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments).)
The analyzer tries to warn you about bad indentation but it may not catch all
edge cases. If the indentation is wrong, the analyzer may not recognize some
sections, such as `Args:`, which may then lead to further errors about missing
documentation for the parameters.

#### What to document [inconsistent-docstring]

<a name="inconsistent-docstring"></a>
The analyzer **requires docstrings** for:

*   each **module** (.bzl file)
*   each **public function** (i.e. a function not starting with an underscore)
    that contains at least 5 statements

If a **function has a multi-line docstring**, you also have to document (in
order):

*   all **parameters** of the function (in declaration order)
*   the **return value** if the function returns a value, i.e. it contains
    `return foo` instead of just `return`
*   a **deprecation warning** if the function is deprecated, cf. the
    [deprecation section](#deprecated-symbol) above

### Naming conventions

<a name="name-with-wrong-case"></a>
<a name="provider-name-suffix"></a>
<a name="confusing-name"></a>
Categories: [name-with-wrong-case] [provider-name-suffix] [confusing-name]

**TL;DR**: Most Starlark identifiers should be `snake_case`, not `camelCase`.

In detail, the rules are the following:

*   Use **`lower_snake_case`** for:
    *   **Functions**
    *   **Parameters**
    *   **Mutable** variables
*   Use **`UPPER_SNAKE_CASE`** for:
    *   **Constants** that are **immutable**. The variable must not be rebound
        and also its deep contents must not be changed.
*   Use **`UpperCamelCase`** for:
    *   **Providers**. In addition, provider names have to end in the
        **suffix `Info`**.
*   **Never** use:
    *   **Builtin names** like `print`, `True`. Rebinding these is too
        confusing.
    *   **One-letter** names that are easy to confuse, namely **`O`, `l`, `I`**
        (easy to confuse with `0`, `I`, `l`, respectively).
    *   **Multiple underscores**: `__`, `___`, `____`, etc.
*   Use the **underscore `_`**:
    *   **only** to ignore the result of an assignment, as in `a, _ = tuple`.
    *   **never** to read the value of `_`, e.g. `f(_)`.

### Statements without effects [no-effect]

<a name="no-effect"></a>
If a statement is just an expression that is not a function call, the analyzer
warns `expression result not used`. Most likely, you forgot to do something with
that value. Examples: `1 + foo()`, `foo[bar]`.

#### List comprehensions

List comprehensions inside a function should be transformed to a for-loop.
This transformation is not possible at the top level because for-loops are only
allowed inside a function to encourage declarative code at the top level.
Therefore list comprehension at the top level are allowed but you should
consider whether it would be more readable to move them to a function. Example:

```
# This is allowed at the top level (but consider moving it to a function):
[do_something_with(foo) for foo in bar]

def baz():
  # This is BAD inside a function:
  [do_something_with(foo) for foo in bar]
  # Instead, use a for loop:
  for foo in bar:
    do_something_with(foo)
```

### Return value lint [missing-return-value]

<a name="missing-return-value"></a>
If a function returns with a value (`return foo`) in some execution paths and
without one (just `return` or reaching the end of a function) in other execution
paths, the analyzer will warn about this. The reason for this is that you
probably forgot to return the right value in some execution paths. If this is
not the case, you should make your intent clear by writing `return None`
instead.

#### Example

```
def foo():
  if ...:
    return False
  elif ...:
    # do stuff
    # forgot return statement here, which generates the warning
  else:
    return True
```

#### "I know the else-branch cannot happen"

Suppose you have code like this:

```
def foo():
  if cond1:
    return foo
  elif cond2:
    return bar
  # I know the else-branch can't happen but the analyzer complains
```

In such a case, just add `fail("unreachable")` or something along these lines
to the end of the function in order to silence the warning.

### Uninitialized variables [uninitialized-variable]

<a name="uninitialized-variable"></a>
If a variable is not initialized before it's used on every execution path, the
analyzer warns about it:

```
def foo():
  if cond1:
    foo = bar
  elif cond2:
    foo = baz
  print(foo) # warning: 'foo' may not have been initialized
```

Most likely, the author forgot to initialize `foo` in the `else` branch.
However, if this is a false positive because you know that the `else` branch can
never happen, add an `else`-branch with the line `fail("unreachable")` or
something similar. Then the analyzer won't complain.

If your code is more complex and it is impossible to determine statically that a
variable is initialized before usage, just initialize the variable with `None`
before using it.

### Unused bindings [unused-binding]

<a name="unused-binding"></a>
If a binding of an identifier is not used, the analyzer warns about it:

```
_PRIVATE_GLOBAL_UNUSED = 0 # warns because never used
PUBLIC_GLOBAL_UNUSED = 1 # doesn't warn because it might be exported

def public_function_unused(): # doesn't warn because it might be exported
  pass

# warns because unused function and parameter:
def _private_function_unused(param_unused):
  # warns because unused local variable → rename to '_' or  '_foo':
  for foo in range(0, 100):
    pass
```

The analyzer warns about unused identifiers except in the following cases:

*   If the identifier is global and public, there are no warnings because it may
    be `load()`ed from somewhere else.
*   If the identifier is `_`, there is no warning because the underscore signals
    that the a value is ignored.
*   If the identifier is a local variable and starts with an underscore, there
    is no warning because the underscore signals that the value is ignored (as
    opposed to global variables, where the underscore signifies that it is
    private).

#### Silencing the warning

If you want to **silence the warning**, you can do one of the following:

*   **Remove** the unused definition.
*   In case of a local variable, **rename** the variable to start with an
    underscore.
*   Look at the special cases below.

#### Unused parameters with fixed names

In case of a parameter whose name you cannot change (because you have to conform
to some API), you can use the following pattern:

```
def foo(param1, param2):
  _ignore = (param1, param2)
  # or:
  _ = (param1, param2)
  ...
```

This way, the parameters are used for the assignment of the ignored variable
`_ignore`. Hence the analyzer will not warn.

#### Re-exporting `load()`ed names

If you want to `load()` a name just to re-export it (and not use it in the
current file), use the following pattern.


```
# If you just want to re-export 'foo', DON'T do this:
load("bar.bzl", "foo")

# DO this instead:
load("bar.bzl", _foo="foo")
foo = _foo
```

This way, the name is still re-exported but doesn't generate a warning.

### Deprecated API

See [documentation](../skylark/lib/ctx.html) for more information.

### Miscellaneous lints

*   <a name="unreachable-statement"></a>**unreachable statements** [unreachable-statement]
*   <a name="load-at-top"></a>**Load statements** must be **at the top** of the file (after the docstring)
    [load-at-top]

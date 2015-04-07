# Concepts

Skylark is the code name of the extension mechanism. It lets you write custom
build rules as well as compose existing ones into [macros](macros.md).

## Loading a Skylark module

Use the `load` statement to import a symbol from a Skylark module.

```python
load("/build_tools/rules/maprule", "maprule")
```

This code will load the file `build_tools/rules/maprule.bzl` and add the
`maprule` symbol to the environment. This can be used to load new rules,
functions or constants (e.g. a string, a list, etc.). Multiple symbols can be
imported by using additional arguments to the call to `load`. Arguments must
be string literals (no variable) and `load` statements must appear at
top-level, i.e. they cannot be in a function body.

Visibility doesn't affect loading. You don't need to use `exports_files`
to make a Skylark file visible.

Symbols starting with `_` are private and cannot be loaded from other files.

## Macros and rules

A [macro](macros.md) in Skylark is a function that instantiates rules. The
function is evaluated as soon as the BUILD file is read. Bazel has little
information about macros: if your macro generates a `genrule`, Bazel will behave
as if you wrote the `genrule`. As a result, `bazel query` will only list the
generated genrule.

A [rule](rules.md) in Skylark is more powerful than a macro, as it can access
Bazel internals and have full control over what is going on. It may for example
pass information to other rules. A rule defined in Skylark will behave in a
similar way as a native rule.

If a macro becomes complex, it is often a good idea to make it a rule.

## Evaluation model

A build consists of three phases.

* **Loading phase**. First, we load and evaluate all Skylark modules and all BUILD
  files that are needed for the build. The execution of the BUILD files simply
  instantiates rules. This is where macros are evaluated.

* **Analysis phase**. The code of the rules is executed (their `implementation`
  function), and actions are instantiated. An action describes how to generate
  a set of outputs from a set of inputs, e.g. "run gcc on hello.c and get
  hello.o". It is important to note that we have to list explicitly which
  files will be generated before executing the actual commands.

* **Execution phase**. Actions are executed, when at least one of their outputs is
  required. If a file is missing or if a command fails to generate one output,
  the build fails. Tests are run during this phase, as they are actions.

## Language

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



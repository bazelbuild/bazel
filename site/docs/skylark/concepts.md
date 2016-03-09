---
layout: documentation
title: Skylark - Concepts
---
# Concepts

## Loading a Skylark extension

Use the `load` statement to import a symbol from a `.bzl` Skylark
extension.

```python
load("/build_tools/rules/maprule", "maprule")
```

This code will load the file `build_tools/rules/maprule.bzl` and add the
`maprule` symbol to the environment. This can be used to load new rules,
functions or constants (e.g. a string, a list, etc.). Multiple symbols can be
imported by using additional arguments to the call to `load`. Arguments must
be string literals (no variable) and `load` statements must appear at
top-level, i.e. they cannot be in a function body.

`load` also supports aliases, i.e. you can assign different names to the
imported symbols.

```python
load("/build_tools/rules/maprule", maprule_alias = "maprule")
```

You define multiple aliases within one `load` statement. Moreover, the argument
list can contain both aliases and regular symbol names. The following example is
perfectly legal (please note when to use quotation marks).

```python
load("/path/to/my_rules", "some_rule", nice_alias = "some_other_rule")
```

Symbols starting with `_` are private and cannot be loaded from other files.
Visibility doesn't affect loading: you don't need to use `exports_files` to make
a Skylark file visible.

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

* **Loading phase**. First, we load and evaluate all Skylark extensions and all BUILD
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

* Although some data structures are mutable, all objects are recursively frozen
  and become recursively immutable before Bazel invokes Skylark
  and after it is done with such evaluation,
  i.e. when a .bzl file is loaded, when a BUILD file is processed,
  when a Skylark-defined rule is evaluated to create a configured target, or
  when a callback function is called to compute a configured target attribute.
  These objects that are frozen notably include the recursive contents of any
  global variable exported by a .bzl file or imported by a `load()` statement,
  and any parameter passed to a callback function or result returned by it.
  From the point of view of the Skylark code
  in given a Bazel-initiated evaluation,
  objects passed as input or present in the evaluation's initial environment
  are immutable, whereas objects created during the evaluation are mutable.
  From the point of view of the Bazel code that evaluates said Skylark code,
  all inputs and outputs of the evaluation are recursively immutable
  and all evaluations are deterministic,
  which guarantees the hermeticity of the build,
  and allows sharing of evaluations without any fear of side-effects.

* Lists are mutable, but dicts and sets are immutable.
  This is temporary: dicts will be made mutable in the near future;
  however there are no plans to make sets mutable at this time.

* All global values are constant (they cannot be reassigned).

* `x += y` is syntactic sugar for `x = x + y`. Even if `x` and `y` are lists,
  dicts or sets, the original value is not mutated, so references to `x`
  that were assigned before the operation will see the old value. This behavior
  is temporary, and will follow Python semantics in the future.

* The `+` operator is defined for dictionaries, returning an immutable
  concatenated dictionary created from the entries of the original
  dictionaries. In case of duplicate keys, we use values from the second
  operand. If you need compatibility with Python, we suggest this syntax:
  `dict(a.items() + b.items())`.

* Dictionary assignment has slightly different semantics: `d["x"] = y` is
  syntactic sugar for `d = d + {"x": y}` or `d += {"x": y}`. This behavior
  is temporary, and will follow Python semantics in the future.

* Dictionaries have deterministic order when iterating (sorted by key).

* Sets use a custom order when iterating (see
  [documentation](lib/globals.html#set)).

* Recursion is not allowed.

The following Python features are not supported:

* `class` (see [`struct`](lib/globals.html#struct) function)
* `import` (see [`load`](#loading-a-skylark-extension) statement)
* `while`, `yield`
* `lambda`
* `is` (use `==` instead)
* `try`, `raise`, `except`, `finally` (see [`fail`](lib/globals.html#fail)
  for fatal errors).
* most builtin functions, most methods



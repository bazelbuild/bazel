---
layout: documentation
title: Style guide for bzl files
---

# .bzl file style guide

## Style

* When in doubt, follow the
  [Python style guide](https://www.python.org/dev/peps/pep-0008/).

* Code should be documented using
  [docstrings](https://www.python.org/dev/peps/pep-0257/). Use a docstring at
  the top of the file, and a docstring for each public function.

* Variables and function names use lowercase with words separated by underscores
  (`[a-z][a-z0-9_]*`), e.g. `cc_library`. Top-level private values start with
  one underscore. Bazel enforces that private values cannot be used from other
  files. Local variables should not use the underscore prefix.

* As in BUILD files, there is no strict line length limit as labels can be long.
  When possible, try to use at most 79 characters per line.

* In keyword arguments, spaces around the equal sign are optional. In general,
  we follow the BUILD file convention when calling macros and native rules, and
  the Python convention for other functions, e.g.

```python
def fct(name, srcs):
  filtered_srcs = my_filter(source=srcs)
  native.cc_library(
    name = name,
    srcs = filtered_srcs,
  )
```

## Macros

A [macro](macros.md) is a function which instantiates one or many rules during
the loading phase.

* Macros must accept a name attribute and each invocation should specify a name.
  The generated name attribute of rules should include the name attribute as a
  prefix. For example, `my_macro(name = "foo")` can generate a rule `foo` and a
  rule `foo_gen`. *Rationale*: Users should be able to find easily which macro
  generated a rule. Also, automated refactoring tools need a way to identify a
  specific rule to edit.

* When calling a macro, use only keyword arguments. *Rationale*: This is for
  consistency with rules, it greatly improves readability.


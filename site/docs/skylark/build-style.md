---
layout: documentation
title: Style guide for BUILD files
---

# BUILD file style guide

In `BUILD` files, we take the same approach as in Go: We let the machine take care
of most formatting issues.
[Buildifier](https://github.com/bazelbuild/buildifier) is a tool that parses and
emits the source code in a standard style. Every `BUILD` file is therefore
formatted in the same automated way, which makes formatting a non-issue during
code reviews. It also makes it easier for tools to understand, edit, and
generate `BUILD` files.

`BUILD` file formatting must match the output of `buildifier`.

## Formatting example

```python
package(default_visibility = ["//visibility:public"])

py_test(
    name = "foo_test",
    srcs = glob(["*.py"]),
    data = [
        "//data/production/foo:startfoo",
        "//foo",
        "//third_party/java/jdk:jdk-k8",
    ],
    flaky = 1,
    deps = [
        ":check_bar_lib",
        ":foo_data_check",
        ":pick_foo_port",
        "//pyglib",
        "//testing/pybase",
    ],
)
```

## File structure

We recommend to use the following order (every element is optional):

  * Package description (a comment)

  * All `load()` statements

  * The `package()` function.

  * Calls to rules and macros

Buildifier makes a distinction between a standalone comment and a comment
attached to an element. If a comment is not attached to a specific element, use
an empty line after it. The distinction is important when doing automated
changes (e.g. to decide if we keep or remove a comment when we delete a rule).

```python
# Standalone comment (e.g. to make a section in a file)

# Comment for the cc_library below
cc_library(name = "cc")
```

## Conventions

 * Use uppercase and underscores to declare constants (e.g. `GLOBAL_CONSTANT`),
   use lowercase and underscores to declare variables (e.g. `my_variable`).

 * Labels should be canonicalized. Use `//foo/bar` instead of `//foo/bar:bar`.
   Use `:bar` if it is defined in the same package. *Rationale*: It makes clear
   if a label is local to a package. Sorting a list of labels is messy if all
   labels do not use the same conventions.

 * Labels should never be split, even if they are longer than 79 characters.
   Labels should be string literals whenever possible. Rationale: It makes
   find and replace easy. It also improves readability.

 * The value of the name attribute should be a literal constant string (except
   in macros). *Rationale*: External tools use the name attribute to refer a
   rule. They need to find rules without having to interpret code.

## Differences with Python style guide

Although compatibility with
[Python style guide](https://www.python.org/dev/peps/pep-0008/) is a goal, there
are a few differences:

 * No strict line length limit. Long comments and long strings are often split
   to 79 columns, but it is not required. It should not be enforced in code
   reviews or presubmit scripts. *Rationale*: Labels can be long and exceed this
   limit. It is common for `BUILD` files to be generated or edited by tools, which
   does not go well with a line length limit.

 * Implicit string concatenation is not supported. Use the `+` operator.
   *Rationale*: `BUILD` files contain many string lists. It is easy to forget a
   comma, which leads to a complete different result. This has created many bugs
   in the past. [See also this discussion.](https://lwn.net/Articles/551438/)

 * Use spaces around the `=` sign for keywords arguments in rules. *Rationale*:
   Named arguments are much more frequent than in Python and are always on a
   separate line. Spaces improve readability. This convention has been around
   for a long time, and we don't think it is worth modifying all existing
   `BUILD` files.

 * By default, use double quotation marks for strings. *Rationale*: This is not
   specified in the Python style guide, but it recommends consistency. So we
   decided to use only double-quoted strings. Many languages use double-quotes
   for string literals.

 * Use a single blank line between two top-level definitions. *Rationale*: The
   structure of a `BUILD` file is not like a typical Python file. It has only
   top-level statements. Using a single-blank line makes `BUILD` files shorter.

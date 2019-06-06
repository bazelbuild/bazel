---
layout: documentation
title: BUILD Style Guide
---

# BUILD Style Guide


In `BUILD` files, we take the same approach as in Go: We let the machine take care
of most formatting issues.
[Buildifier](https://github.com/bazelbuild/buildifier) is a tool that parses and
emits the source code in a standard style. Every `BUILD` file is therefore
formatted in the same automated way, which makes formatting a non-issue during
code reviews. It also makes it easier for tools to understand, edit, and
generate `BUILD` files.

`BUILD` file formatting must match the output of `buildifier`.


## Contents

- [Formatting example](#formatting-example)
- [File structure](#file-structure)
- [References to targets in the current package](#references-to-targets-in-the-current-package)
- [Target naming](#target-naming)
- [Visibility](#visibility)
- [Dependencies](#dependencies)
- [Globs](#globs)
- [Other conventions](#other-conventions)
- [Differences with Python style guide](#differences-with-python-style-guide)

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
    flaky = True,
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

## References to targets in the current package

Files should be referred to by their paths relative to the package directory
(without ever using up-references, such as `..`). Generated files should be
prefixed with "`:`" to indicate that they are not sources. Source files
should not be prefixed with `:`. Rules should be prefixed with `:`. For
example, assuming `x.cc` is a source file:

```python
cc_library(
    name = "lib",
    srcs = ["x.cc"],
    hdrs = [":gen-header"],
)

genrule(
    name = "gen-header",
    srcs = [],
    outs = ["x.h"],
    cmd = "echo 'int x();' > $@",
)
```

## Target naming

Target names should be descriptive. If a target contains one source file,
the target should generally be named after that source (e.g., a `cc_library`
for `chat.cc` should be named "`chat`").

The eponymous target for a package (the target with the same name as the
containing directory) should provide the functionality described by the
directory name. If there is no such target, do not create an eponymous
target.

Prefer using the short name when referring to an eponymous target (`//x`
instead of `//x:x`). If you are in the same package, prefer the local
reference (`:x` instead of `//x`).

## Visibility

Visibility should be scoped as tightly as possible, while still allowing access
by tests and reverse dependencies. Use `__pkg__` and `__subpackages__` as
appropriate.

Avoid setting package `default_visibility` to `//visibility:public`.
`//visibility:public` should be individually set only for targets in the
project's public API. These could be libraries that are designed to be depended
on by external projects or binaries that could be used by an external project's
build process.

## Dependencies

Dependencies should be restricted to direct dependencies (dependencies
needed by the sources listed in the rule). Do not list transitive
dependencies.

Package-local dependencies should be listed first and referred to in a way
compatible with the
[References to targets in the current package](#references-to-targets-in-the-current-package)
section above (not by their absolute package name).

Prefer to list dependencies directly, as a single list. Putting the "common"
dependencies of several targets into a variable reduces maintainability, makes
it impossible for tools to change the dependencies of a target and can lead to
unused dependencies.

## Globs

Indicate "no targets" with `[]`. Do not use a glob that matches nothing: it
is more error-prone and less obvious than an empty list.

### Recursive

Do not use recursive globs (for example, `glob(["**/*.java"])`).

Recursive globs make BUILD files difficult to reason about because they skip
subdirectories containing BUILD files.

Recursive globs are generally less efficient than having a BUILD file per
directory with a dependency graph defined between them as this enables better
forge caching and parallelism.

We recommend authoring a BUILD file per directory and defining a dependency
graph between them instead.

### Non-recursive

Non-recursive globs are generally acceptable.

## Other conventions

 * Use uppercase and underscores to declare constants (e.g. `GLOBAL_CONSTANT`),
   use lowercase and underscores to declare variables (e.g. `my_variable`).

 * Labels should never be split, even if they are longer than 79 characters.
   Labels should be string literals whenever possible. Rationale: It makes
   find and replace easy. It also improves readability.

 * The value of the name attribute should be a literal constant string (except
   in macros). *Rationale*: External tools use the name attribute to refer a
   rule. They need to find rules without having to interpret code.

 * When setting boolean-type attributes, use boolean values, not integer values.
   For legacy reasons, rules will still convert integers to booleans as needed,
   but this is discouraged. *Rationale*: `flaky = 1` could be misread as saying
   "deflake this target by rerunning it once". `flaky = True` unambiguously says
   "this test is flaky".

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

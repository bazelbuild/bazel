---
layout: documentation
title: .bzl Style Guide
---

# .bzl Style Guide


[Starlark](language.md) is a language that defines how software is built, and as
such it is both a programming and a configuration language.

You will use Starlark to write BUILD files, macros, and build rules. Macros and
rules are essentially meta-languages - they define how BUILD files are written.
BUILD files are intended to be simple and repetitive.

All software is read more often than it is written. This is especially true for
Starlark, as engineers read BUILD files to understand dependencies of their
targets and details of their builds.This reading will often happen in passing,
in a hurry, or in parallel to accomplishing some other task. Consequently,
simplicity and readability are very important so that users can parse and
comprehend BUILD files quickly.

When a user opens a BUILD file, they quickly want to know the list of targets in
the file; or review the list of sources of that C++ library; or remove a
dependency from that Java binary. Each time you add a layer of abstraction, you
make it harder for a user to do these tasks.

BUILD files are also analyzed and updated by many different tools. Tools may not
be able to edit your BUILD file if it uses abstractions. Keeping your BUILD
files simple will allow you to get better tooling. As a code base grows, it
becomes more and more frequent to do changes across many BUILD files in order to
update a library or do a cleanup.

Do not create a macro just to avoid some amount of repetition in BUILD files.
The [DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) principle
doesn’t really apply here. The goal is not to make the file shorter; the goal is
to make your files easy to process, both by humans and tools.

## General advice

*   Use [skylint](skylint.md).
*   Follow [testing guidelines](testing.md).

## Style


### Python style

When in doubt, follow the
[Python style guide](https://www.python.org/dev/peps/pep-0008/). In particular,
use 4 spaces for indentation (we previously recommended 2, but we now follow the
Python convention).

### Docstring

Document files and functions using [docstrings](skylint.md#docstrings). Use a
docstring at the top of each `.bzl` file, and a docstring for each public
function.

### Document rules and aspects

Rules and aspects, along with their attributes, as well as providers and their
fields, should be documented using the `doc` argument.

### Naming convention

*   Variables and function names use lowercase with words separated by
    underscores (`[a-z][a-z0-9_]*`), e.g. `cc_library`.
*   Top-level private values start with one underscore. Bazel enforces that
    private values cannot be used from other files. Local variables should not
    use the underscore prefix.

### Line length

As in BUILD files, there is no strict line length limit as labels can be long.
When possible, try to use at most 79 characters per line.

### Keyword arguments

In keyword arguments, spaces around the equal sign are optional, but be
consistent within any given call. In general, we follow the BUILD file
convention when calling macros and native rules, and the Python convention for
other functions, e.g.

```python
def fct(name, srcs):
    filtered_srcs = my_filter(source = srcs)
    native.cc_library(
        name = name,
        srcs = filtered_srcs,
        testonly = True,
    )
```

### Boolean values

Prefer values `True` and `False` (rather than of `1` and `0`) for boolean values
(e.g. when using a boolean attribute in a rule).

### Use print only for debugging

Do not use the `print()` function in production code; it is only intended for
debugging, and will spam all direct and indirect users of your `.bzl` file. The
only exception is that you may submit code that uses `print()` if it is disabled
by default and can only be enabled by editing the source -- for example, if all
uses of `print()` are guarded by `if DEBUG:` where `DEBUG` is hardcoded to
`False`. Be mindful of whether these statements are useful enough to justify
their impact on readability.


## Macros

A macro is a function which instantiates one or more rules during the loading
phase. In general, use rules whenever possible instead of macros. The build
graph seen by the user is not the same as the one used by Bazel during the
build - macros are expanded _before Bazel does any build graph analysis._

Because of this, when something goes wrong, the user will need to understand
your macro’s implementation to troubleshoot build problems. Additionally, `bazel
query` results can be hard to interpret because targets shown in the results
come from macro expansion. Finally, aspects are not aware of macros, so tooling
depending on aspects (IDEs and others) might fail.

A safe use for macros is leaf nodes, such as macros defining test permutations:
in that case, only the "end users" of those targets need to know about those
additional nodes, and any build problems introduced by macros are never far from
their usage.

For macros that define non-leaf nodes, follow these best practices:

*   A macro should take a `name` argument and define a target with that name.
    That target becomes that macro's _main target_.
*   All other targets defined by a macro should have their names preceded with a
    `_`, include the `name` attribute as a prefix, and have restricted
    visibility.
*   All the targets created in the macro should be coupled in some way to the
    main target.
*   Keep the parameter names in the macro consistent. If a parameter is passed
    as an attribute value to the main target, keep its name the same. If a macro
    parameter serves the same purpose as a common rule attribute, such as
    `deps`, name as you would the attribute (see below).
*   When calling a macro, use only keyword arguments. This is consistent with
    rules, and greatly improves readability.

Engineers often write macros when the Starlark API of relevant rules is
insufficient for their specific use case, regardless of whether the rule is
defined within Bazel in native code, or in Starlark. If you’re facing this
problem, ask the rule author if they can extend the API to accomplish your
goals.

As a rule of thumb, the more macros resemble the rules, the better.

## Rules

*   Rules, aspects, and their attributes should use lower_case names (“snake
    case”).
*   Rule names are nouns that describe the main kind of artifact produced by the
    rule, from the point of view of its dependencies (or for leaf rules, the
    user). This is not necessarily a file suffix. For instance, a rule that
    produces C++ artifacts meant to be used as Python extensions might be called
    `py_extension`. For most languages, typical rules include:
    *   `*_library` - a compilation unit or "module".
    *   `*_binary` - a target producing an executable or a deployment unit.
    *   `*_test` - a test target. This can include multiple tests. Expect all
        tests in a `*_test` target to be variations on the same theme, for
        example, testing a single library.
    *   `*_import`: a target encapsulating a pre-compiled artifact, such as a
        `.jar`, or a `.dll` that is used during compilation.
*   Use consistent names and types for attributes. Some generally applicable
    attributes include:
    *   `srcs`: `label_list`, allowing files: source files, typically
        human-authored.
    *   `deps`: `label_list`, typically _not_ allowing files: compilation
        dependencies.
    *   `data`: `label_list`, allowing files: data files, such as test data etc.
    *   `runtime_deps`: `label_list`: runtime dependencies that are not needed
        for compilation.
*   For any attributes with non-obvious behavior (for example, string templates
    with special substitutions, or tools that are invoked with specific
    requirements), provide documentation using the `doc` keyword argument to the
    attribute's declaration (`attr.label_list()` or similar).
*   Rule implementation functions should almost always be private functions
    (named with a leading underscore). A common style is to give the
    implementation function for `myrule` the name `_myrule_impl`.
*   Pass information between your rules using a well-defined
    [provider](rules.md#providers) interface. Declare and document provider
    fields.
*   Design your rule with extensibility in mind. Consider that other rules might
    want to interact with your rule, access your providers, and reuse the
    actions you create.
*   Follow [performance guidelines](performance.md) in your rules.


Project: /_project.yaml
Book: /_book.yaml

# .bzl style guide

{% include "_buttons.html" %}

This page covers basic style guidelines for Starlark and also includes
information on macros and rules.

[Starlark](/rules/language) is a
language that defines how software is built, and as such it is both a
programming and a configuration language.

You will use Starlark to write `BUILD` files, macros, and build rules. Macros and
rules are essentially meta-languages - they define how `BUILD` files are written.
`BUILD` files are intended to be simple and repetitive.

All software is read more often than it is written. This is especially true for
Starlark, as engineers read `BUILD` files to understand dependencies of their
targets and details of their builds. This reading will often happen in passing,
in a hurry, or in parallel to accomplishing some other task. Consequently,
simplicity and readability are very important so that users can parse and
comprehend `BUILD` files quickly.

When a user opens a `BUILD` file, they quickly want to know the list of targets in
the file; or review the list of sources of that C++ library; or remove a
dependency from that Java binary. Each time you add a layer of abstraction, you
make it harder for a user to do these tasks.

`BUILD` files are also analyzed and updated by many different tools. Tools may not
be able to edit your `BUILD` file if it uses abstractions. Keeping your `BUILD`
files simple will allow you to get better tooling. As a code base grows, it
becomes more and more frequent to do changes across many `BUILD` files in order to
update a library or do a cleanup.

Important: Do not create a variable or macro just to avoid some amount of
repetition in `BUILD` files. Your `BUILD` file should be easily readable both by
developers and tools. The
[DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself){: .external} principle doesn't
really apply here.

## General advice {:#general-advice}

*   Use [Buildifier](https://github.com/bazelbuild/buildtools/tree/master/buildifier#linter){: .external}
    as a formatter and linter.
*   Follow [testing guidelines](/rules/testing).

## Style {:#style}

### Python style {:#python-style}

When in doubt, follow the
[PEP 8 style guide](https://www.python.org/dev/peps/pep-0008/) where possible.
In particular, use four rather than two spaces for indentation to follow the
Python convention.

Since
[Starlark is not Python](/rules/language#differences-with-python),
some aspects of Python style do not apply. For example, PEP 8 advises that
comparisons to singletons be done with `is`, which is not an operator in
Starlark.


### Docstring {:#docstring}

Document files and functions using [docstrings](https://github.com/bazelbuild/buildtools/blob/master/WARNINGS.md#function-docstring){: .external}.
Use a docstring at the top of each `.bzl` file, and a docstring for each public
function.

### Document rules and aspects {:#doc-rules-aspects}

Rules and aspects, along with their attributes, as well as providers and their
fields, should be documented using the `doc` argument.

### Naming convention {:#naming-convention}

*   Variables and function names use lowercase with words separated by
    underscores (`[a-z][a-z0-9_]*`), such as `cc_library`.
*   Top-level private values start with one underscore. Bazel enforces that
    private values cannot be used from other files. Local variables should not
    use the underscore prefix.

### Line length {:#line-length}

As in `BUILD` files, there is no strict line length limit as labels can be long.
When possible, try to use at most 79 characters per line (following Python's
style guide, [PEP 8](https://www.python.org/dev/peps/pep-0008/)). This guideline
should not be enforced strictly: editors should display more than 80 columns,
automated changes will frequently introduce longer lines, and humans shouldn't
spend time splitting lines that are already readable.

### Keyword arguments {:#keyword-arguments}

In keyword arguments, spaces around the equal sign are preferred:

```python
def fct(name, srcs):
    filtered_srcs = my_filter(source = srcs)
    native.cc_library(
        name = name,
        srcs = filtered_srcs,
        testonly = True,
    )
```

### Boolean values {:#boolean-values}

Prefer values `True` and `False` (rather than of `1` and `0`) for boolean values
(such as when using a boolean attribute in a rule).

### Use print only for debugging {:#print-for-debugging}

Do not use the `print()` function in production code; it is only intended for
debugging, and will spam all direct and indirect users of your `.bzl` file. The
only exception is that you may submit code that uses `print()` if it is disabled
by default and can only be enabled by editing the source -- for example, if all
uses of `print()` are guarded by `if DEBUG:` where `DEBUG` is hardcoded to
`False`. Be mindful of whether these statements are useful enough to justify
their impact on readability.

## Macros {:#macros}

A macro is a function which instantiates one or more rules during the loading
phase. In general, use rules whenever possible instead of macros. The build
graph seen by the user is not the same as the one used by Bazel during the
build - macros are expanded *before Bazel does any build graph analysis.*

Because of this, when something goes wrong, the user will need to understand
your macro's implementation to troubleshoot build problems. Additionally, `bazel
query` results can be hard to interpret because targets shown in the results
come from macro expansion. Finally, aspects are not aware of macros, so tooling
depending on aspects (IDEs and others) might fail.

A safe use for macros is for defining additional targets intended to be
referenced directly at the Bazel CLI or in BUILD files: In that case, only the
*end users* of those targets need to know about them, and any build problems
introduced by macros are never far from their usage.

For macros that define generated targets (implementation details of the macro
which are not supposed to be referred to at the CLI or depended on by targets
not instantiated by that macro), follow these best practices:

*   A macro should take a `name` argument and define a target with that name.
    That target becomes that macro's *main target*.
*   Generated targets, that is all other targets defined by a macro, should:
    *   Have their names prefixed by `<name>` or `_<name>`. For example, using
        `name = '%s_bar' % (name)`.
    *   Have restricted visibility (`//visibility:private`), and
    *   Have a `manual` tag to avoid expansion in wildcard targets (`:all`,
        `...`, `:*`, etc).
*   The `name` should only be used to derive names of targets defined by the
    macro, and not for anything else. For example, don't use the name to derive
    a dependency or input file that is not generated by the macro itself.
*   All the targets created in the macro should be coupled in some way to the
    main target.
*   Conventionally, `name` should be the first argument when defining a macro.
*   Keep the parameter names in the macro consistent. If a parameter is passed
    as an attribute value to the main target, keep its name the same. If a macro
    parameter serves the same purpose as a common rule attribute, such as
    `deps`, name as you would the attribute (see below).
*   When calling a macro, use only keyword arguments. This is consistent with
    rules, and greatly improves readability.

Engineers often write macros when the Starlark API of relevant rules is
insufficient for their specific use case, regardless of whether the rule is
defined within Bazel in native code, or in Starlark. If you're facing this
problem, ask the rule author if they can extend the API to accomplish your
goals.

As a rule of thumb, the more macros resemble the rules, the better.

See also [macros](/extending/macros#conventions).

## Rules {:#rules}

*   Rules, aspects, and their attributes should use lower_case names ("snake
    case").
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
    *   `deps`: `label_list`, typically *not* allowing files: compilation
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
    [provider](/extending/rules#providers) interface. Declare and document provider
    fields.
*   Design your rule with extensibility in mind. Consider that other rules might
    want to interact with your rule, access your providers, and reuse the
    actions you create.
*   Follow [performance guidelines](/rules/performance) in your rules.

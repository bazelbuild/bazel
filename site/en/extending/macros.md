Project: /_project.yaml
Book: /_book.yaml
{# disableFinding("Currently") #}
{# disableFinding(TODO) #}

# Macros

This page covers the basics of using macros and includes typical use cases,
debugging, and conventions.

A macro is a function called from the `BUILD` file that can instantiate rules.
Macros are mainly used for encapsulation and code reuse of existing rules and
other macros.

Macros come in two flavors: symbolic macros, which are described on this page,
and [legacy macros](legacy-macros.md). Where possible, we recommend using
symbolic macros for code clarity.

Symbolic macros offer typed arguments (string to label conversion, relative to
where the macro was called) and the ability to restrict and specify the
visibility of targets created. They are designed to be amenable to lazy
evaluation (which will be added in a future Bazel release). Symbolic macros are
available by default in Bazel 8. Where this document mentions `macros`, it's
referring to **symbolic macros**.

## Usage {:#usage}

Macros are defined in `.bzl` files by calling the `macro()` function with two
parameters: `attrs` and `implementation`.

### Attributes {:#attributes}

`attrs` accepts a dictionary of attribute name to [attribute
types](https://bazel.build/rules/lib/toplevel/attr#members), which represents
the arguments to the macro. Two common attributes - name and visibility - are
implicitly added to all macros and are not included in the dictionary passed to
attrs.

```starlark
# macro/macro.bzl
my_macro = macro(
    attrs = {
        "deps": attr.label_list(mandatory = True, doc = "The dependencies passed to the inner cc_binary and cc_test targets"),
        "create_test": attr.bool(default = False, configurable = False, doc = "If true, creates a test target"),
    },
    implementation = _my_macro_impl,
)
```

Attribute type declarations accept the
[parameters](https://bazel.build/rules/lib/toplevel/attr#parameters),
`mandatory`, `default`, and `doc`. Most attribute types also accept the
`configurable` parameter, which determines wheher the attribute accepts
`select`s. If an attribute is `configurable`, it will parse non-`select` values
as an unconfigurable `select` - `"foo"` will become
`select({"//conditions:default": "foo"})`. Learn more in [selects](#selects).

### Implementation {:#implementation}

`implementation` accepts a function which contains the logic of the macro.
Implementation functions often create targets by calling one or more rules, and
they are are usually private (named with a leading underscore). Conventionally,
they are named the same as their macro, but prefixed with `_` and suffixed with
`_impl`.

Unlike rule implementation functions, which take a single argument (`ctx`) that
contains a reference to the attributes, macro implementation functions accept a
parameter for each argument.

```starlark
# macro/macro.bzl
def _my_macro_impl(name, deps, create_test):
    cc_library(
        name = name + "_cc_lib",
        deps = deps,
    )

    if create_test:
        cc_test(
            name = name + "_test",
            srcs = ["my_test.cc"],
            deps = deps,
        )
```

### Declaration {:#declaration}

Macros are declared by loading and calling their definition in a `BUILD` file.

```starlark

# pkg/BUILD

my_macro(
    name = "macro_instance",
    deps = ["src.cc"] + select(
        {
            "//config_setting:special": ["special_source.cc"],
            "//conditions:default": [],
        },
    ),
    create_tests = True,
)
```

This would create targets
`//pkg:macro_instance_cc_lib` and`//pkg:macro_instance_test`.

## Details {:#usage-details}

### naming conventions for targets created {:#naming}

The names of any targets or submacros created by a symbolic macro must
either match the macro's `name` parameter or must be prefixed by `name` followed
by `_` (preferred), `.` or `-`. For example, `my_macro(name = "foo")` may only
create files or targets named `foo`, or prefixed by `foo_`, `foo-` or `foo.`,
for example, `foo_bar`.

Targets or files that violate macro naming convention can be declared, but
cannot be built and cannot be used as dependencies.

Non-macro files and targets within the same package as a macro instance should
*not* have names that conflict with potential macro target names, though this
exclusivity is not enforced. We are in the progress of implementing
[lazy evaluation](#laziness) as a performance improvement for Symbolic macros,
which will be impaired in packages that violate the naming schema.

### restrictions {:#restrictions}

Symbolic macros have some additional restrictions compared to legacy macros.

Symbolic macros

*   must take a `name` argument and a `visibility` argument
*   must have an `implementation` function
*   may not return values
*   may not mutate their `args`
*   may not call `native.existing_rules()` unless they are special `finalizer`
    macros
*   may not call `native.package()`
*   may not call `glob()`
*   may not call `native.environment_group()`
*   must create targets whose names adhere to the [naming schema](#naming)
*   can't refer to input files that weren't declared or passed in as an argument
    (see [visibility](#visibility) for more details).

### Visibility {:#visibility}

TODO: Expand this section

#### Target visibility {:#target-visibility}

At default, targets created by symbolic macros are visible to the package in
which they are created. They also accept a `visibility` attribute, which can
expand that visibility to the caller of the macro (by passing the `visibility`
attribute directly from the macro call to the target created) and to other
packages (by explicitly specifying them in the target's visibility).

#### Dependency visibility {:#dependency-visibility}

Macros must have visibility to the files and targets they refer to. They can do
so in one of the following ways:

*   Explicitly passed in as an `attr` value to the macro

```starlark

# pkg/BUILD
my_macro(... deps = ["//other_package:my_tool"] )
```

*   Implicit default of an `attr` value

```starlark
# my_macro:macro.bzl
my_macro = macro(
  attrs = {"deps" : attr.label_list(default = ["//other_package:my_tool"])} )
```

*   Already visible to the macro definition

```starlark
# other_package/BUILD
cc_binary(
    name = "my_tool",
    visibility = "//my_macro:\\__pkg__",
)
```

### Selects {:#selects}

If an attribute is `configurable`, then the macro implementation function will
always see the attribute value as `select`-valued. For example, consider the
following macro:

```starlark
my_macro = macro(
    attrs = {"deps": attr.label_list()},  # configurable unless specified otherwise
    implementation = _my_macro_impl,
)
```

If `my_macro` is invoked with `deps = ["//a"]`, that will cause `_my_macro_impl`
to be invoked with its `deps` parameter set to `select({"//conditions:default":
["//a"]})`.

Rule targets reverse this transformation, and store trivial `select`s as their
unconditional values; in this example, if `_my_macro_impl` declares a rule
target `my_rule(..., deps = deps)`, that rule target's `deps` will be stored as
`["//a"]`.

## Finalizers {:#finalizers}

A rule finalizer is a special symbolic macro which - regardless of its lexical
position in a BUILD file - is evaluated in the final stage of loading a package,
after all non-finalizer targets have been defined. Unlike ordinary symbolic
macros, a finalizer can call `native.existing_rules()`, where it behaves
slightly differently than in legacy macros: it only returns the set of
non-finalizer rule targets. The finalizer may assert on the state of that set or
define new targets.

To declare a finalizer, call `macro()` with `finalizer = True`:

```starlark
def _my_finalizer_impl(name, visibility, tags_filter):
    for r in native.existing_rules().values():
        for tag in r.get("tags", []):
            if tag in tags_filter:
                my_test(
                    name = name + "_" + r["name"] + "_finalizer_test",
                    deps = [r["name"]],
                    data = r["srcs"],
                    ...
                )
                continue

my_finalizer = macro(
    attrs = {"tags_filter": attr.string_list(configurable = False)},
    implementation = _impl,
    finalizer = True,
)
```

## Laziness {:#laziness}

IMPORTANT: We are in the process of implementing lazy macro expansion and
evaluation. This feature is not available yet.

Currently, all macros are evaluated as soon as the BUILD file is loaded, which
can negatively impact performance for targets in packages that also have costly
unrelated macros. In the future, non-finalizer symbolic macros will only be
evaluated if they're required for the build. The prefix naming schema helps
Bazel determine which macro to expand given a requested target.

## Migration troubleshooting {:#troubleshooting}

Here are some common migration headaches and how to fix them.

*   Legacy macro calls `glob()`

Move the `glob()` call to your BUILD file (or to a legacy macro called from the
BUILD file), and pass the `glob()` value to the symbolic macro using a
label-list attribute:

```starlark
# BUILD file
my_macro(
    ...,
    deps = glob(...),
)
```

*   Legacy macro has a parameter that isn't a valid starlark `attr` type.

Pull as much logic as possible into a nested symbolic macro, but keep the
top level macro a legacy macro.

*  Legacy macro calls a rule that creates a target that breaks the naming schema

That's okay, just don't depend on the "offending" target. The naming check will
be quietly ignored.


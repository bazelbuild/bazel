---
layout: documentation
title: Rules
category: extending
---

# Rules

A **rule** defines a series of [**actions**](#actions) that Bazel performs on
inputs to produce a set of outputs, which are referenced in
[**providers**](#providers) returned by the rule's
[**implementation function**](#implementation-function). For example, a C++
binary rule might:

1.  Take a set of `.cpp` source files (inputs).
2.  Run `g++` on the source files (action).
3.  Return the `DefaultInfo` provider with the executable output and other files
    to make available at runtime.
4.  Return the `CcInfo` provider with C++-specific information gathered from the
    target and its dependencies.

From Bazel's perspective, `g++` and the standard C++ libraries are also inputs
to this rule. As a rule writer, you must consider not only the user-provided
inputs to a rule, but also all of the tools and libraries required to execute
the actions.

Before creating or modifying any rule, ensure you are familiar with Bazel's
[build phases](concepts.md). It will be important to understand the three phases
of a build (loading, analysis and execution). It will also be useful to learn
about [macros](macros.md) to understand the difference between rules and macros.
To get started, we recommend that you first follow the
[Rules Tutorial](rules-tutorial.md). The current page can be used as a
reference.

A few rules are built into Bazel itself. These *native rules*, such as
`cc_library` and `java_binary`, provide some core support for certain languages.
By defining your own rules, you can add similar support for languages and tools
that Bazel does not support natively.

Bazel provides an extensibility model for writing rules using the
[Starlark](language.md) language. These rules are written in `.bzl` files, which
can be loaded directly from `BUILD` files.

When defining your own rule, you get to decide what attributes it supports and
how it generates its outputs.

The rule's `implementation` function defines its exact behavior during the
[analysis phase](concepts.md#evaluation-model). This function does not run any
external commands. Rather, it registers [actions](#actions) that will be used
later during the execution phase to build the rule's outputs, if they are
needed.

## Rule creation

In a `.bzl` file, use the [rule](lib/globals.html#rule) function to define a new
rule, and store the result in a global variable. The call to `rule` specifies
[attributes](#attributes) and an
[implementation function](#implementation-function):

```python
example_library = rule(
    implementation = _example_library_impl,
    attrs = {
        "deps": attr.label_list(),
        ...
    },
)
```

This defines a [kind of rule](../query.html#kind) named `example_library`.

The call to `rule` also must specify if the rule creates an
[executable](#executable-rules) output (with `executable=True`), or specifically
a test executable (with `test=True`). If the latter, the rule is a *test rule*,
and the name of the rule must end in `_test`.

## Target instantiation

Rules can be [loaded](../build-ref.html#load) and called in `BUILD` files:

```python
load('//some/pkg:rules.bzl', 'example_library')

example_library(
    name = "example_target",
    deps = [":another_target"],
    ...
)
```

Each call to a build rule returns no value, but has the side effect of defining
a target. This is called *instantiating* the rule. This specifies a name for the
new target and values for the target's [attributes](#attributes).

Rules can also be called from Starlark functions and loaded in `.bzl` files.
Starlark functions that call rules are called [Starlark macros](macros.md).
Starlark macros must ultimately be called from `BUILD` files, and can only be
called during the [loading phase](concepts.md#evaluation-model), when `BUILD`
files are evaluated to instantiate targets.

## Attributes

An *attribute* is a rule argument. Attributes can provide specific values to a
target's [implementation](#implementation-function), or they can refer to other
targets, creating a graph of dependencies.

Rule-specific attributes, such as `srcs` or `deps`, are defined by passing a map
from attribute names to schemas (created using the [`attr`](lib/attr.html)
module) to the `attrs` parameter of `rule`.
[Common attributes](../be/common-definitions.html#common-attributes), such as
`name` and `visibility`, are implicitly added to all rules. Additional
attributes are implicitly added to
[executable and test rules](#executable-rules) specifically. Attributes which
are implicitly added to a rule cannot be included in the dictionary passed to
`attrs`.

### Dependency attributes

Rules that process source code usually define the following attributes to handle
various [types of dependencies](../build-ref.html#types_of_dependencies):

*   `srcs` specifies source files processed by a target's actions. Often, the
    attribute schema specifies which file extensions are expected for the sort
    of source file the rule processes. Rules for languages with header files
    generally specify a separate `hdrs` attribute for headers processed by a
    target and its consumers.
*   `deps` specifies code dependencies for a target. The attribute schema should
    specify which [providers](#providers) those dependencies must provide. (For
    example, `cc_library` provides `CcInfo`.)
*   `data` specifies files to be made available at runtime to any executable
    which depends on a target. That should allow arbitrary files to be
    specified.

```python
example_library = rule(
    implementation = _example_library_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = [".example"]),
        "hdrs": attr.label_list(allow_files = [".header"]),
        "deps": attr.label_list(providers = [ExampleInfo]),
        "data": attr.label_list(allow_files = True),
        ...
    },
)
```

These are examples of *dependency attributes*. Any attribute definied with
[`attr.label_list`](lib/attr.html#label_list) (or
[`attr.label`](lib/attr.html#label)) specifies dependencies of a certain type
between a target and the targets whose labels (or the corresponding
[`Label`](lib/Label.html) objects) are listed in that attribute when the target
is defined. The repository, and possibly the path, for these labels is resolved
relative to the defined target.

```python
example_library(
    name = "my_target",
    deps = [":other_target"],
)

example_library(
    name = "other_target",
    ...
)
```

In this example, `other_target` is a dependency of `my_target`, and therefore
`other_target` is analyzed first. It is an error if there is a cycle in the
dependency graph of targets.

<a name="private-attributes"></a>

### Private attributes and implicit dependencies

A dependency attribute with a default value creates an *implicit dependency*. It
is implicit because it's a part of the target graph that the user does not
specify in a BUILD file. Implicit dependencies are useful for hard-coding a
relationship between a rule and a *tool* (a build-time dependency, such as a
compiler), since most of the time a user is not interested in specifying what
tool the rule uses. Inside the rule's implementation function, this is treated
the same as other dependencies.

If you want to provide an implicit dependency without allowing the user to
override that value, you can make the attribute *private* by giving it a name
that begins with an underscore (`_`). Private attributes must have default
values. It generally only makes sense to use private attributes for implicit
dependencies.

```python
example_library = rule(
    implementation = _example_library_impl,
    attrs = {
        ...
        "_compiler": attr.label(
            default = Label("//tools:example_compiler"),
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
    },
)
```

In this example, every target of type `example_library` will have an implicit
dependency on the compiler `//tools:example_compiler`. This allows
`example_library`'s implementation function to generate actions that invoke the
compiler, even though the user did not pass its label as an input. Since
`_compiler` is a private attribute, we know for sure that `ctx.attr._compiler`
will always point to `//tools:example_compiler` in all targets of this rule
type. Alternatively, we could have named the attribute `compiler` without the
underscore and kept the default value. That would let users substitute a
different compiler if necessary, but it requires no awareness of the compiler's
label otherwise.

Implicit dependencies are generally used for tools that reside in the same
repository as the rule implementation. If the tool comes from the
[execution platform](../platforms.html) or a different repository instead, the
rule should obtain that tool from a [toolchain](../toolchains.html).

### Output attributes

*Output attributes*, such as [`attr.output`](lib/attr.html#output) and
[`attr.output_list`](lib/attr.html#output_list), declare an output file that the
target generates. This differs from the dependency attributes in two ways:

*   It defines an output file target instead of referring to a target defined
    elsewhere.
*   The output file target depends on the instantiated rule target, instead of
    the other way around.

Typically, output attributes are only used when a rule needs to create outputs
with user-defined names which cannot be based on the target name. If a rule has
one output attribute, it is typically named `out` or `outs`.

Output attributes are the preferred way of creating *predeclared outputs*, which
can be specifically depended upon or
[requested at the command line](#requesting-output-files).

## Implementation function

Every rule requires an `implementation` function. These functions are executed
strictly in the [analysis phase](concepts.md#evaluation-model) and transform the
graph of targets generated in the loading phase into a graph of
[actions](#actions) to be performed during the execution phase. As such,
implementation functions can not actually read or write files.

Rule implementation functions are usually private (named with a leading
underscore). Conventionally, they are named the same as their rule, but suffixed
with `_impl`.

Implementation functions take exactly one parameter: a
[rule context](lib/ctx.html), conventionally named `ctx`. They return a list of
[providers](#providers).

### Targets

Dependencies are represented at analysis time as [`Target`](lib/Target.html)
objects. These objects contain the [providers](#providers) generated when the
target's implementation function was executed.

[`ctx.attr`](lib/ctx.html#attr) has fields corresponding to the names of each
dependency attribute, containing `Target` objects representing each direct
dependency via that attribute. For `label_list` attributes, this is a list of
`Targets`. For `label` attributes, this is a single `Target` or `None`.

A list of provider objects are returned by a target's implementation function:

```python
return [ExampleInfo(headers = depset(...))]
```

Those can be accessed using index notation (`[]`), with the type of provider as
a key. These can be [custom providers](#custom-providers) defined in Starlark or
[providers for native rules](lib/skylark-provider.html) available as Starlark
global variables.

For example, if a rule takes header files via a `hdrs` attribute and provides
them to the compilation actions of the target and its consumers, it could
collect them like so:

```python
def _example_library_impl(ctx):
    ...
    transitive_headers = [dep[ExampleInfo].headers for dep in ctx.attr.deps]
```

For the legacy style in which a [`struct`](lib/struct.html) is returned from a
target's implementation function instead of a list of provider objects:

```python
return struct(example_info = struct(headers = depset(...)))
```

Providers can be retrieved from the corresponding field of the `Target` object:

```python
transitive_headers = [dep.example_info.headers for dep in ctx.attr.deps]
```

This style is strongly discouraged and rules should be
[migrated away from it](#migrating-from-legacy-providers).

### Files

Files are represented by [`File`](lib/File.html) objects. Since Bazel does not
perform file I/O during the analysis phase, these objects cannot be used to
directly read or write file content. Rather, they are passed to action-emitting
functions (see [`ctx.actions`](lib/actions.html)) to construct pieces of the
action graph.

A `File` can either be a source file or a generated file. Each generated file
must be an output of exactly one action. Source files cannot be the output of
any action.

For each dependency attribute, the corresponding field of
[`ctx.files`](lib/ctx.html#files) contains a list of the default outputs of all
dependencies via that attribute:

```python
def _example_library_impl(ctx):
    ...
    headers = depset(ctx.files.hdrs, transitive=transitive_headers)
    srcs = ctx.files.srcs
    ...
```

[`ctx.file`](lib/ctx.html#file) contains a single `File` or `None` for
dependency attributes whose specs set `allow_single_file=True`.
[`ctx.executable`](`ctx.executable`) behaves the same as `ctx.file`, but only
contains fields for dependency attributes whose specs set `executable=True`.

### Declaring outputs

During the analysis phase, a rule's implementation function can create outputs.
Since all labels have to be known during the loading phase, these additional
outputs have no labels. `File` objects for outputs can be created using using
[`ctx.actions.declare_file`](lib/actions.html#declare_file) and
[`ctx.actions.declare_directory`](lib/actions.html#declare_directory). Often,
the names of outputs are based on the target's name,
[`ctx.label.name`](lib/ctx.html#label):

```python
def _example_library_impl(ctx):
  ...
  output_file = ctx.actions.declare_file(ctx.label.name + ".output")
  ...
```

For *predeclared outputs*, like those created for
[output attributes](#output-attributes), `File` objects instead can be retrieved
from the corresponding fields of [`ctx.outputs`](lib/ctx.html#outputs).

### Actions

An action describes how to generate a set of outputs from a set of inputs, for
example "run gcc on hello.c and get hello.o". When an action is created, Bazel
doesn't run the command immediately. It registers it in a graph of dependencies,
because an action can depend on the output of another action. For example, in C,
the linker must be called after the compiler.

General-purpose functions that create actions are defined in
[`ctx.actions`](lib/actions.html):

*   [`ctx.actions.run`](lib/actions.html#run), to run an executable.
*   [`ctx.actions.run_shell`](lib/actions.html#run_shell), to run a shell
    command.
*   [`ctx.actions.write`](lib/actions.html#write), to write a string to a file.
*   [`ctx.actions.expand_template`](lib/actions.html#expand_template), to
    generate a file from a template.

[`ctx.actions.args`](lib/actions.html#args) can be used to efficiently
accumulate the arguments for actions. It avoids flattening depsets until
execution time:

```python
def _example_library_impl(ctx):
    ...

    transitive_headers = [dep.example_info.headers for dep in ctx.attr.deps]
    headers = depset(ctx.files.hdrs, transitive=transitive_headers)
    srcs = ctx.files.srcs
    inputs = depset(srcs, transitive=[headers])
    output_file = ctx.actions.declare_file(ctx.label.name + ".output")

    args = ctx.actions.args()
    args.add_joined("-h", headers, join_with=",")
    args.add_joined("-s", srcs, join_with=",")
    args.add("-o", output_file)

    ctx.actions.run(
        mnemonic="ExampleCompile",
        executable = ctx.executable._compiler,
        arguments=args,
        inputs = inputs,
        outputs = [output_file],
    )
    ...
```

Actions take a list or depset of input files and generate a (non-empty) list of
output files. The set of input and output files must be known during the
[analysis phase](concepts.md#evaluation-model). It might depend on the value of
attributes, including providers from dependencies, but it cannot depend on the
result of the execution. For example, if your action runs the unzip command, you
must specify which files you expect to be inflated (before running unzip).
Actions which create a variable number of files internally can wrap those in a
single file (e.g. a zip, tar, or other archive format).

Actions must list all of their inputs. Listing inputs that are not used is
permitted, but inefficient.

Actions must create all of their outputs. They may write other files, but
anything not in outputs will not be available to consumers. All declared outputs
must be written by some action.

Actions are comparable to pure functions: They should depend only on the
provided inputs, and avoid accessing computer information, username, clock,
network, or I/O devices (except for reading inputs and writing outputs). This is
important because the output will be cached and reused.

Dependencies are resolved by Bazel, which will decide which actions are
executed. It is an error if there is a cycle in the dependency graph. Creating
an action does not guarantee that it will be executed, that depends on whether
its outputs are needed for the build.

### Providers

Providers are pieces of information that a rule exposes to other rules that
depend on it. This data can include output files, libraries, parameters to pass
on a tool's command line, or anything else a target's consumers should know
about.

Since a rule's implementation function can only read providers from the
instantiated target's immediate dependencies, rules need to forward any
information from a target's dependencies that needs to be known by a target's
consumers, generally by accumulating that into a [`depset`](lib/depset.html).

A target's providers are specified by a list of `Provider` objects returned by
the implementation function.

Old implementation functions can also be written in a legacy style where the
implementation function returns a [`struct`](lib/struct.html) instead of list of
provider objects. This style is strongly discouraged and rules should be
[migrated away from it](#migrating-from-legacy-providers).

#### Default outputs

A target's *default outputs* are the outputs that are requested by default when
the target is requested for build at the command line. For example, a
`java_library` target `//pkg:foo` has `foo.jar` as a default output, so that
will be built by the command `bazel build //pkg:foo`.

Default outputs are specified by the `files` parameter of
[`DefaultInfo`](lib/DefaultInfo.html):

```python
def _example_library_impl(ctx):
    ...
    return [
        DefaultInfo(files = depset([output_file]), ...),
        ...
    ]
```

If `DefaultInfo` is not returned by a rule implementation or the `files`
parameter is not specified, `DefaultInfo.files` defaults to all
[*predeclared outputs*](#predeclared-outputs) (generally, those created by
[output attributes](#output-attributes)).

Rules that perform actions should provide default outputs, even if those outputs
are not expected to be directly used. Actions that are not in the graph of the
requested outputs are pruned. If an output is only used by a target's consumers,
those actions will not be performed when the target is built in isolation. This
makes debugging more difficult because rebuilding just the failing target won't
reproduce the failure.

#### Runfiles

Runfiles are a set of files used by a target at runtime (as opposed to build
time). During the [execution phase](concepts.md#evaluation-model), Bazel creates
a directory tree containing symlinks pointing to the runfiles. This stages the
environment for the binary so it can access the runfiles during runtime.

Runfiles can be added manually during rule creation.
[`runfiles`](lib/runfiles.html) objects can be created by the `runfiles` method
on the rule context, [`ctx.runfiles`](lib/ctx.html#runfiles) and passed to the
`runfiles` parameter on `DefaultInfo`. The executable output of
[executable rules](#executable-rules) is implicitly added to the runfiles.

Some rules specify attributes, generally named
[`data`](../be/common-definitions.html#common.data), whose outputs are added to
a targets' runfiles. Runfiles should also be merged in from any targets which
provide runtime dependencies (including `deps` and `data`) or source files
(which might include `filegroup` targets with associated `data`):

```python
def _example_library_impl(ctx):
    ...
    runfiles = ctx.runfiles(files = ctx.files.data)
    all_targets = ctx.attr.srcs + ctx.attr.hdrs + ctx.attr.deps + ctx.attr.data
    runfiles = runfiles.merge_all([
        target[DefaultInfo].default_runfiles
        for target in all_targets
    ])
    return [
        DefaultInfo(..., runfiles = runfiles),
        ...
    ]
```

#### Coverage configuration

Since rules can only convey information about their immediate dependencies, all
rules need to return the `InstrumentedFilesInfo` provider in order for the
`coverage` command to collect data about any transitive dependencies.

That provider can be created with
[`coverage_common.instrumented_files_info`](lib/coverage_common.html#instrumented_files_info).
The `dependency_attributes` parameter of `instrumented_files_info` should list
all runtime dependency attributes, including code dependencies like `deps` and
data dependencies like `data`. The `source_attributes` parameter should list the
rule's source files attributes if coverage instrumentation
[is added at build time](#code-coverage-instrumentation), or if it might be
added at runtime (for interpreted languages):

```python
def _example_library_impl(ctx):
    ...
    return [
        ...
        coverage_common.instrumented_files_info(
            dependency_attributes = ["deps", "data"],
            # Omitted if coverage is not supported for this rule:
            source_attributes = ["srcs", "hdrs"],
        )
        ...
    ]
```

#### Custom providers

Providers can be defined using the [`provider`](lib/globals.html#provider)
function to convey rule-specific information:

```python
ExampleInfo = provider(
    "Info needed to compile/link Example code.",
    fields={
        "headers": "depset of header Files from transitive dependencies.",
        "files_to_link": "depset of Files from compilation.",
    })
```

Rule implementation functions can then construct and return provider instances:

```python
def _example_library_impl(ctx):
  ...
  return [
      ...
      ExampleInfo(
          headers = headers,
          files_to_link = depset(
              [output_file],
              transitive = [
                  dep[ExampleInfo].files_to_link for dep in ctx.attr.deps
              ],
          ),
      )
  ]
```

<a name="executable-rules"></a>

## Executable rules and test rules

Executable rules define targets that can be invoked by a `bazel run` command.
Test rules are a special kind of executable rule whose targets can also be
invoked by a `bazel test` command. Executable and test rules are created by
setting the respective [`executable`](lib/globals.html#rule.executable) or
[`test`](lib/globals.html#rule.test) argument to `True` in the call to `rule`:

```python
example_binary = rule(
   implementation = _example_binary_impl,
   executable = True,
   ...
)

example_test = rule(
   implementation = _example_binary_impl,
   test = True,
   ...
)
```

Test rules must have names that end in `_test`. (Test *target* names also often
end in `_test` by convention, but this is not required.) Non-test rules must not
have this suffix.

Both kinds of rules must produce an executable output file (which may or may not
be predeclared) that will be invoked by the `run` or `test` commands. To tell
Bazel which of a rule's outputs to use as this executable, pass it as the
`executable` argument of a returned [`DefaultInfo`](lib/DefaultInfo.html)
provider. That `executable` is added to the default outputs of the rule (so you
don't need to pass that to both `executable` and `files`). It's also implicitly
added to the [runfiles](#runfiles):

```python
def _example_binary_impl(ctx):
    executable = ctx.actions.declare_file(ctx.label.name)
    ...
    return [
        DefaultInfo(executable = executable, ...),
        ...
    ]
```

The action that generates this file must set the executable bit on the file. For
a [`ctx.actions.run`](lib/actions.html#run) or
[`ctx.actions.run_shell`](lib/actions.html#run_shell) action this should be done
by the underlying tool that is invoked by the action. For a
[`ctx.actions.write`](lib/actions.html#write) action, pass `is_executable=True`.

As [legacy behavior](#deprecated-predeclared-outputs), executable rules have a
special `ctx.outputs.executable` predeclared output. This file serves as the
default executable if you do not specify one using `DefaultInfo`; it must not be
used otherwise. This output mechanism is deprecated because it does not support
customizing the executable file's name at analysis time.

See examples of an
[executable rule](https://github.com/bazelbuild/examples/blob/master/rules/executable/fortune.bzl)
and a
[test rule](https://github.com/bazelbuild/examples/blob/master/rules/test_rule/line_length.bzl).

[Executable rules](be/common-definitions.html#common-attributes-binaries) and
[test rules](be/common-definitions.html#common-attributes-tests) have additional
attributes implicitly defined, in addition to those added for
[all rules](be/common-definitions.html#common-attributes). The defaults of
implicitly-added attributes cannot be changed, though this can be worked around
by wrapping a private rule in a [Starlark macro](macros.md) which alters the
default:

```python
def example_test(size="small", **kwargs):
  _example_test(size=size, **kwargs)

_example_test = rule(
 ...
)
```

### Runfiles location

When an executable target is run with `bazel run` (or `test`), the root of the
runfiles directory is adjacent to the executable. The paths relate as follows:

```python
# Given executable_file and runfile_file:
runfiles_root = executable_file.path + ".runfiles"
workspace_name = ctx.workspace_name
runfile_path = runfile_file.short_path
execution_root_relative_path = "%s/%s/%s" % (
    runfiles_root, workspace_name, runfile_path)
```

The path to a `File` under the runfiles directory corresponds to
[`File.short_path`](lib/File.html#short_path).

The binary executed directly by `bazel` is adjacent to the root of the
`runfiles` directory. However, binaries called *from* the runfiles can't make
the same assumption. To mitigate this, each binary should provide a way to
accept its runfiles root as a parameter using an environment or command line
argument/flag. This allows binaries to pass the correct canonical runfiles root
to the binaries it calls. If that's not set, a binary can guess that it was the
first binary called and look for an adjacent runfiles directory.

## Advanced topics

### Requesting output files

A single target can have several output files. When a `bazel build` command is
run, some of the outputs of the targets given to the command are considered to
be *requested*. Bazel only builds these requested files and the files that they
directly or indirectly depend on. (In terms of the action graph, Bazel only
executes the actions that are reachable as transitive dependencies of the
requested files.)

In addition to [default outputs](#default-outputs), any *predeclared output* can
be explicitly requested on the command line. Rules can specify predeclared
outputs via [output attributes](#output-attributes). In that case, the user
explicitly chooses labels for outputs when they instantiate the rule. To obtain
[`File`](lib/File.html) objects for output attributes, use the corresponding
attribute of [`ctx.outputs`](lib/ctx.html#outputs). Rules can
[implicitly define predeclared outputs](#deprecated-predeclared-outputs) based
on the target name as well, but this feature is deprecated.

In addition to default outputs, there are *output groups*, which are collections
of output files that may be requested together. These can be requested with
[`--output_groups`](../command-line-reference.html#flag--output_groups). For
example, if a target `//pkg:mytarget` is of a rule type that has a `debug_files`
output group, these files can be built by running `bazel build //pkg:mytarget
--output_groups=debug_files`. Since non-predeclared outputs don't have labels,
they can only be requested by appearing in the default outputs or an output
group.

Output groups can be specified with the
[`OutputGroupInfo`](lib/OutputGroupInfo.html) provider. Note that unlike many
built-in providers, `OutputGroupInfo` can take parameters with arbitrary names
to define output groups with that name:

```python
def _example_library_impl(ctx):
    ...
    debug_file = ctx.actions.declare_file(name + ".pdb")
    ...
    return [
        DefaultInfo(files = depset([output_file]), ...),
        OutputGroupInfo(
            debug_files = depset([debug_file]),
            all_files = depset([output_file, debug_file]),
        ),
        ...
    ]
```

Note that `OutputGroupInfo` generally shouldn't be used to convey specific sorts
of files from a target to the actions of its consumers. Define
[rule-specific providers](#custom-providers) for that instead.

### Configurations

Imagine that you want to build a C++ binary for a different architecture. The
build can be complex and involve multiple steps. Some of the intermediate
binaries, like compilers and code generators, have to run on
[the execution platform](../platforms.html#overview) (which could be your host,
or a remote executor). Some binaries like the final output must be built for the
target architecture.

For this reason, Bazel has a concept of "configurations" and transitions. The
topmost targets (the ones requested on the command line) are built in the
"target" configuration, while tools that should run on the execution platform
are built in an "exec" configuration. Rules may generate different actions based
on the configuration, for instance to change the cpu architecture that is passed
to the compiler. In some cases, the same library may be needed for different
configurations. If this happens, it will be analyzed and potentially built
multiple times.

By default, Bazel builds a target's dependencies in the same configuration as
the target itself, in other words without transitions. When a dependency is a
tool that's needed to help build the target, the corresponding attribute should
specify a transition to an exec configuration. This causes the tool and all its
dependencies to build for the execution platform.

For each dependency attribute, you can use `cfg` to decide if dependencies
should build in the same configuration or transition to an exec configuration.
If a dependency attribute has the flag `executable=True`, `cfg` must be set
explicitly. This is to guard against accidentally building a tool for the wrong
configuration.
[See example](https://github.com/bazelbuild/examples/blob/master/rules/actions_run/execute.bzl)

In general, sources, dependent libraries, and executables that will be needed at
runtime can use the same configuration.

Tools that are executed as part of the build (e.g., compilers, code generators)
should be built for an exec configuration. In this case, specify `cfg="exec"` in
the attribute.

Otherwise, executables that are used at runtime (e.g. as part of a test) should
be built for the target configuration. In this case, specify `cfg="target"` in
the attribute.

`cfg="target"` doesn't actually do anything: it's purely a convenience value to
help rule designers be explicit about their intentions. When `executable=False`,
which means `cfg` is optional, only set this when it truly helps readability.

You can also use `cfg=my_transition` to use
[user-defined transitions](config.html#user-defined-transitions), which allow
rule authors a great deal of flexibility in changing configurations, with the
drawback of
[making the build graph larger and less comprehensible](config.html#memory-and-performance-considerations).

**Note**: Historically, Bazel didn't have the concept of execution platforms,
and instead all build actions were considered to run on the host machine.
Because of this, there is a single "host" configuration, and a "host" transition
that can be used to build a dependency in the host configuration. Many rules
still use the "host" transition for their tools, but this is currently
deprecated and being migrated to use "exec" transitions where possible.

There are numerous differences between the "host" and "exec" configurations:

*   "host" is terminal, "exec" isn't: Once a dependency is in the "host"
    configuration, no more transitions are allowed. You can keep making further
    configuration transitions once you're in an "exec" configuration.
*   "host" is monolithic, "exec" isn't: There is only one "host" configuration,
    but there can be a different "exec" configuration for each execution
    platform.
*   "host" assumes you run tools on the same machine as Bazel, or on a
    significantly similar machine. This is no longer true: you can run build
    actions on your local machine, or on a remote executor, and there's no
    guarantee that the remote executor is the same CPU and OS as your local
    machine.

Both the "exec" and "host" configurations apply the same option changes, (i.e.,
set `--compilation_mode` from `--host_compilation_mode`, set `--cpu` from
`--host_cpu`, etc). The difference is that the "host" configuration starts with
the **default** values of all other flags, whereas the "exec" configuration
starts with the **current** values of flags, based on the target configuration.

<a name="fragments"></a>

### Configuration fragments

Rules may access
[configuration fragments](lib/skylark-configuration-fragment.html) such as
`cpp`, `java` and `jvm`. However, all required fragments must be declared in
order to avoid access errors:

```python
def _impl(ctx):
    # Using ctx.fragments.cpp would lead to an error since it was not declared.
    x = ctx.fragments.java
    ...

my_rule = rule(
    implementation = _impl,
    fragments = ["java"],      # Required fragments of the target configuration
    host_fragments = ["java"], # Required fragments of the host configuration
    ...
)
```

`ctx.fragments` only provides configuration fragments for the target
configuration. If you want to access fragments for the host configuration, use
`ctx.host_fragments` instead.

### Runfiles symlinks

Normally, the relative path of a file in the runfiles tree is the same as the
relative path of that file in the source tree or generated output tree. If these
need to be different for some reason, you can specify the `root_symlinks` or
`symlinks` arguments. The `root_symlinks` is a dictionary mapping paths to
files, where the paths are relative to the root of the runfiles directory. The
`symlinks` dictionary is the same, but paths are implicitly prefixed with the
name of the workspace.

```python
    ...
    runfiles = ctx.runfiles(
        root_symlinks = {"some/path/here.foo": ctx.file.some_data_file2}
        symlinks = {"some/path/here.bar": ctx.file.some_data_file3}
    )
    # Creates something like:
    # sometarget.runfiles/
    #     some/
    #         path/
    #             here.foo -> some_data_file2
    #     <workspace_name>/
    #         some/
    #             path/
    #                 here.bar -> some_data_file3
```

If `symlinks` or `root_symlinks` is used, be careful not to map two different
files to the same path in the runfiles tree. This will cause the build to fail
with an error describing the conflict. To fix, you will need to modify your
`ctx.runfiles` arguments to remove the collision. This checking will be done for
any targets using your rule, as well as targets of any kind that depend on those
targets. This is especially risky if your tool is likely to be used transitively
by another tool; symlink names must be unique across the runfiles of a tool and
all of its dependencies.

### Code coverage instrumentation

If a rule implementation adds coverage instrumentation at build time, it needs
to account for that in its implementation function.
[ctx.coverage_instrumented](lib/ctx.html#coverage_instrumented) returns true in
coverage mode if a target's sources should be instrumented:

```python
# Are this rule's sources instrumented?
if ctx.coverage_instrumented():
  # Do something to turn on coverage for this compile action
```

That same logic governs whether files provided to that target via attributes
listed in `source_attributes` are included in coverage data output.

Logic that always needs to be on in coverage mode (whether a target's sources
specifically are instrumented or not) can be conditioned on
[ctx.configuration.coverage_enabled](lib/configuration.html#coverage_enabled).

If the rule directly includes sources from its dependencies before compilation
(e.g. header files), it may also need to turn on compile-time instrumentation if
the dependencies' sources should be instrumented:

```python
# Are this rule's sources or any of the sources for its direct dependencies
# in deps instrumented?
if (ctx.configuration.coverage_enabled and
    (ctx.coverage_instrumented() or
     any([ctx.coverage_instrumented(dep) for dep in ctx.attr.deps]))):
    # Do something to turn on coverage for this compile action
```

## Deprecated features

### Deprecated predeclared outputs

There are two **deprecated** ways of using predeclared outputs:

*   The [`outputs`](lib/globals.html#rule.outputs) parameter of `rule` specifies
    a mapping between output attribute names and string templates for generating
    predeclared output labels. Prefer using non-predeclared outputs and
    explicitly adding outputs to `DefaultInfo.files`. Use the rule target's
    label as input for rules which consume the output instead of a predeclared
    output's label.

*   For [executable rules](#executable-rules), `ctx.outputs.executable` refers
    to a predeclared executable output with the same name as the rule target.
    Prefer declaring the output explicitly, for example with
    `ctx.actions.declare_file(ctx.label.name)`, and ensure that the command that
    generates the executable sets its permissions to allow execution. Explicitly
    pass the executable output to the `executable` parameter of `DefaultInfo`.

### Runfiles features to avoid

[`ctx.runfiles`](lib/ctx.html#runfiles) and the [`runfiles`](lib/runfiles.html)
type have a complex set of features, many of which are kept for legacy reasons.
We make the following recommendations to reduce complexity:

*   **Avoid** use of the `collect_data` and `collect_default` modes of
    [`ctx.runfiles`](lib/ctx.html#runfiles). These modes implicitly collect
    runfiles across certain hardcoded dependency edges in confusing ways.
    Instead, add files using the `files` or `transitive_files` parameters of
    `ctx.runfiles`, or by merging in runfiles from dependencies with
    `runfiles = runfiles.merge(dep[DefaultInfo].default_runfiles)`.

*   **Avoid** use of the `data_runfiles` and `default_runfiles` of the
    `DefaultInfo` constructor. Specify `DefaultInfo(runfiles = ...)` instead.
    The distinction between "default" and "data" runfiles is maintained for
    legacy reasons. For example, some rules put their default outputs in
    `data_runfiles`, but not `default_runfiles`. Instead of using
    `data_runfiles`, rules should *both* include default outputs and merge in
    `default_runfiles` from attributes which provide runfiles (often
    [`data`](../be/common-definitions.html#common-attributes.data)).

*   When retrieving `runfiles` from `DefaultInfo` (generally only for merging
    runfiles between the current rule and its dependencies), use
    `DefaultInfo.default_runfiles`, **not** `DefaultInfo.data_runfiles`.

### Migrating from legacy providers

Historically, Bazel providers were simple fields on the `Target` object. They
were accessed using the dot operator, and they were created by putting the field
in a struct returned by the rule's implementation function.

*This style is deprecated and should not be used in new code;* see below for
information that may help you migrate. The new provider mechanism avoids name
clashes. It also supports data hiding, by requiring any code accessing a
provider instance to retrieve it using the provider symbol.

For the moment, legacy providers are still supported. A rule can return both
legacy and modern providers as follows:

```python
def _old_rule_impl(ctx):
  ...
  legacy_data = struct(x="foo", ...)
  modern_data = MyInfo(y="bar", ...)
  # When any legacy providers are returned, the top-level returned value is a
  # struct.
  return struct(
      # One key = value entry for each legacy provider.
      legacy_info = legacy_data,
      ...
      # Additional modern providers:
      providers = [modern_data, ...])
```

If `dep` is the resulting `Target` object for an instance of this rule, the
providers and their contents can be retrieved as `dep.legacy_info.x` and
`dep[MyInfo].y`.

In addition to `providers`, the returned struct can also take several other
fields that have special meaning (and thus do not create a corresponding legacy
provider):

*   The fields `files`, `runfiles`, `data_runfiles`, `default_runfiles`, and
    `executable` correspond to the same-named fields of
    [`DefaultInfo`](lib/DefaultInfo.html). It is not allowed to specify any of
    these fields while also returning a `DefaultInfo` provider.

*   The field `output_groups` takes a struct value and corresponds to an
    [`OutputGroupInfo`](lib/OutputGroupInfo.html).

In [`provides`](lib/globals.html#rule.provides) declarations of rules, and in
[`providers`](lib/attr.html#label_list.providers) declarations of dependency
attributes, legacy providers are passed in as strings and modern providers are
passed in by their `*Info` symbol. Be sure to change from strings to symbols
when migrating. For complex or large rule sets where it is difficult to update
all rules atomically, you may have an easier time if you follow this sequence of
steps:

1.  Modify the rules that produce the legacy provider to produce both the legacy
    and modern providers, using the above syntax. For rules that declare they
    return the legacy provider, update that declaration to include both the
    legacy and modern providers.

2.  Modify the rules that consume the legacy provider to instead consume the
    modern provider. If any attribute declarations require the legacy provider,
    also update them to instead require the modern provider. Optionally, you can
    interleave this work with step 1 by having consumers accept/require either
    provider: Test for the presence of the legacy provider using
    `hasattr(target, 'foo')`, or the new provider using `FooInfo in target`.

3.  Fully remove the legacy provider from all rules.

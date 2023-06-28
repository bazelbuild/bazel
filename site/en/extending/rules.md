Project: /_project.yaml
Book: /_book.yaml

# Rules

{% include "_buttons.html" %}

A **rule** defines a series of [**actions**](#actions) that Bazel performs on
inputs to produce a set of outputs, which are referenced in
[**providers**](#providers) returned by the rule's
[**implementation function**](#implementation_function). For example, a C++
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
[build phases](/extending/concepts). It is important to understand the three
phases of a build (loading, analysis, and execution). It is also useful to
learn about [macros](/extending/macros) to understand the difference between rules and
macros. To get started, first review the [Rules Tutorial](/rules/rules-tutorial).
Then, use this page as a reference.

A few rules are built into Bazel itself. These *native rules*, such as
`cc_library` and `java_binary`, provide some core support for certain languages.
By defining your own rules, you can add similar support for languages and tools
that Bazel does not support natively.

Bazel provides an extensibility model for writing rules using the
[Starlark](/rules/language) language. These rules are written in `.bzl` files, which
can be loaded directly from `BUILD` files.

When defining your own rule, you get to decide what attributes it supports and
how it generates its outputs.

The rule's `implementation` function defines its exact behavior during the
[analysis phase](/extending/concepts#evaluation-model). This function does not run any
external commands. Rather, it registers [actions](#actions) that will be used
later during the execution phase to build the rule's outputs, if they are
needed.

## Rule creation

In a `.bzl` file, use the [rule](/rules/lib/globals/bzl#rule) function to define a new
rule, and store the result in a global variable. The call to `rule` specifies
[attributes](#attributes) and an
[implementation function](#implementation_function):

```python
example_library = rule(
    implementation = _example_library_impl,
    attrs = {
        "deps": attr.label_list(),
        ...
    },
)
```

This defines a [kind of rule](/query/language#kind) named `example_library`.

The call to `rule` also must specify if the rule creates an
[executable](#executable-rules) output (with `executable=True`), or specifically
a test executable (with `test=True`). If the latter, the rule is a *test rule*,
and the name of the rule must end in `_test`.

## Target instantiation

Rules can be [loaded](/concepts/build-files#load) and called in `BUILD` files:

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
Starlark functions that call rules are called [Starlark macros](/extending/macros).
Starlark macros must ultimately be called from `BUILD` files, and can only be
called during the [loading phase](/extending/concepts#evaluation-model), when `BUILD`
files are evaluated to instantiate targets.

## Attributes

An *attribute* is a rule argument. Attributes can provide specific values to a
target's [implementation](#implementation_function), or they can refer to other
targets, creating a graph of dependencies.

Rule-specific attributes, such as `srcs` or `deps`, are defined by passing a map
from attribute names to schemas (created using the [`attr`](/rules/lib/toplevel/attr)
module) to the `attrs` parameter of `rule`.
[Common attributes](/reference/be/common-definitions#common-attributes), such as
`name` and `visibility`, are implicitly added to all rules. Additional
attributes are implicitly added to
[executable and test rules](#executable-rules) specifically. Attributes which
are implicitly added to a rule cannot be included in the dictionary passed to
`attrs`.

### Dependency attributes

Rules that process source code usually define the following attributes to handle
various [types of dependencies](/concepts/dependencies#types_of_dependencies):

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

These are examples of *dependency attributes*. Any attribute that specifies
an input label (those defined with
[`attr.label_list`](/rules/lib/toplevel/attr#label_list),
[`attr.label`](/rules/lib/toplevel/attr#label), or
[`attr.label_keyed_string_dict`](/rules/lib/toplevel/attr#label_keyed_string_dict))
specifies dependencies of a certain type
between a target and the targets whose labels (or the corresponding
[`Label`](/rules/lib/builtins/Label) objects) are listed in that attribute when the target
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
specify in a `BUILD` file. Implicit dependencies are useful for hard-coding a
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

In this example, every target of type `example_library` has an implicit
dependency on the compiler `//tools:example_compiler`. This allows
`example_library`'s implementation function to generate actions that invoke the
compiler, even though the user did not pass its label as an input. Since
`_compiler` is a private attribute, it follows that `ctx.attr._compiler`
will always point to `//tools:example_compiler` in all targets of this rule
type. Alternatively, you can name the attribute `compiler` without the
underscore and keep the default value. This allows users to substitute a
different compiler if necessary, but it requires no awareness of the compiler's
label.

Implicit dependencies are generally used for tools that reside in the same
repository as the rule implementation. If the tool comes from the
[execution platform](/extending/platforms) or a different repository instead, the
rule should obtain that tool from a [toolchain](/extending/toolchains).

### Output attributes

*Output attributes*, such as [`attr.output`](/rules/lib/toplevel/attr#output) and
[`attr.output_list`](/rules/lib/toplevel/attr#output_list), declare an output file that the
target generates. These differ from dependency attributes in two ways:

*   They define output file targets instead of referring to targets defined
    elsewhere.
*   The output file targets depend on the instantiated rule target, instead of
    the other way around.

Typically, output attributes are only used when a rule needs to create outputs
with user-defined names which cannot be based on the target name. If a rule has
one output attribute, it is typically named `out` or `outs`.

Output attributes are the preferred way of creating *predeclared outputs*, which
can be specifically depended upon or
[requested at the command line](#requesting_output_files).

## Implementation function

Every rule requires an `implementation` function. These functions are executed
strictly in the [analysis phase](/extending/concepts#evaluation-model) and transform the
graph of targets generated in the loading phase into a graph of
[actions](#actions) to be performed during the execution phase. As such,
implementation functions can not actually read or write files.

Rule implementation functions are usually private (named with a leading
underscore). Conventionally, they are named the same as their rule, but suffixed
with `_impl`.

Implementation functions take exactly one parameter: a
[rule context](/rules/lib/builtins/ctx), conventionally named `ctx`. They return a list of
[providers](#providers).

### Targets

Dependencies are represented at analysis time as [`Target`](/rules/lib/builtins/Target)
objects. These objects contain the [providers](#providers) generated when the
target's implementation function was executed.

[`ctx.attr`](/rules/lib/builtins/ctx#attr) has fields corresponding to the names of each
dependency attribute, containing `Target` objects representing each direct
dependency via that attribute. For `label_list` attributes, this is a list of
`Targets`. For `label` attributes, this is a single `Target` or `None`.

A list of provider objects are returned by a target's implementation function:

```python
return [ExampleInfo(headers = depset(...))]
```

Those can be accessed using index notation (`[]`), with the type of provider as
a key. These can be [custom providers](#custom_providers) defined in Starlark or
[providers for native rules](/rules/lib/providers) available as Starlark
global variables.

For example, if a rule takes header files via a `hdrs` attribute and provides
them to the compilation actions of the target and its consumers, it could
collect them like so:

```python
def _example_library_impl(ctx):
    ...
    transitive_headers = [hdr[ExampleInfo].headers for hdr in ctx.attr.hdrs]
```

For the legacy style in which a [`struct`](/rules/lib/builtins/struct) is returned from a
target's implementation function instead of a list of provider objects:

```python
return struct(example_info = struct(headers = depset(...)))
```

Providers can be retrieved from the corresponding field of the `Target` object:

```python
transitive_headers = [hdr.example_info.headers for hdr in ctx.attr.hdrs]
```

This style is strongly discouraged and rules should be
[migrated away from it](#migrating_from_legacy_providers).

### Files

Files are represented by [`File`](/rules/lib/builtins/File) objects. Since Bazel does not
perform file I/O during the analysis phase, these objects cannot be used to
directly read or write file content. Rather, they are passed to action-emitting
functions (see [`ctx.actions`](/rules/lib/builtins/actions)) to construct pieces of the
action graph.

A `File` can either be a source file or a generated file. Each generated file
must be an output of exactly one action. Source files cannot be the output of
any action.

For each dependency attribute, the corresponding field of
[`ctx.files`](/rules/lib/builtins/ctx#files) contains a list of the default outputs of all
dependencies via that attribute:

```python
def _example_library_impl(ctx):
    ...
    headers = depset(ctx.files.hdrs, transitive=transitive_headers)
    srcs = ctx.files.srcs
    ...
```

[`ctx.file`](/rules/lib/builtins/ctx#file) contains a single `File` or `None` for
dependency attributes whose specs set `allow_single_file=True`.
[`ctx.executable`](/rules/lib/builtins/ctx#executable) behaves the same as `ctx.file`, but only
contains fields for dependency attributes whose specs set `executable=True`.

### Declaring outputs

During the analysis phase, a rule's implementation function can create outputs.
Since all labels have to be known during the loading phase, these additional
outputs have no labels. `File` objects for outputs can be created using
[`ctx.actions.declare_file`](/rules/lib/builtins/actions#declare_file) and
[`ctx.actions.declare_directory`](/rules/lib/builtins/actions#declare_directory). Often,
the names of outputs are based on the target's name,
[`ctx.label.name`](/rules/lib/builtins/ctx#label):

```python
def _example_library_impl(ctx):
  ...
  output_file = ctx.actions.declare_file(ctx.label.name + ".output")
  ...
```

For *predeclared outputs*, like those created for
[output attributes](#output_attributes), `File` objects instead can be retrieved
from the corresponding fields of [`ctx.outputs`](/rules/lib/builtins/ctx#outputs).

### Actions

An action describes how to generate a set of outputs from a set of inputs, for
example "run gcc on hello.c and get hello.o". When an action is created, Bazel
doesn't run the command immediately. It registers it in a graph of dependencies,
because an action can depend on the output of another action. For example, in C,
the linker must be called after the compiler.

General-purpose functions that create actions are defined in
[`ctx.actions`](/rules/lib/builtins/actions):

*   [`ctx.actions.run`](/rules/lib/builtins/actions#run), to run an executable.
*   [`ctx.actions.run_shell`](/rules/lib/builtins/actions#run_shell), to run a shell
    command.
*   [`ctx.actions.write`](/rules/lib/builtins/actions#write), to write a string to a file.
*   [`ctx.actions.expand_template`](/rules/lib/builtins/actions#expand_template), to
    generate a file from a template.

[`ctx.actions.args`](/rules/lib/builtins/actions#args) can be used to efficiently
accumulate the arguments for actions. It avoids flattening depsets until
execution time:

```python
def _example_library_impl(ctx):
    ...

    transitive_headers = [dep[ExampleInfo].headers for dep in ctx.attr.deps]
    headers = depset(ctx.files.hdrs, transitive=transitive_headers)
    srcs = ctx.files.srcs
    inputs = depset(srcs, transitive=[headers])
    output_file = ctx.actions.declare_file(ctx.label.name + ".output")

    args = ctx.actions.args()
    args.add_joined("-h", headers, join_with=",")
    args.add_joined("-s", srcs, join_with=",")
    args.add("-o", output_file)

    ctx.actions.run(
        mnemonic = "ExampleCompile",
        executable = ctx.executable._compiler,
        arguments = [args],
        inputs = inputs,
        outputs = [output_file],
    )
    ...
```

Actions take a list or depset of input files and generate a (non-empty) list of
output files. The set of input and output files must be known during the
[analysis phase](/extending/concepts#evaluation-model). It might depend on the value of
attributes, including providers from dependencies, but it cannot depend on the
result of the execution. For example, if your action runs the unzip command, you
must specify which files you expect to be inflated (before running unzip).
Actions which create a variable number of files internally can wrap those in a
single file (such as a zip, tar, or other archive format).

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
consumers, generally by accumulating that into a [`depset`](/rules/lib/builtins/depset).

A target's providers are specified by a list of `Provider` objects returned by
the implementation function.

Old implementation functions can also be written in a legacy style where the
implementation function returns a [`struct`](/rules/lib/builtins/struct) instead of list of
provider objects. This style is strongly discouraged and rules should be
[migrated away from it](#migrating_from_legacy_providers).

#### Default outputs

A target's *default outputs* are the outputs that are requested by default when
the target is requested for build at the command line. For example, a
`java_library` target `//pkg:foo` has `foo.jar` as a default output, so that
will be built by the command `bazel build //pkg:foo`.

Default outputs are specified by the `files` parameter of
[`DefaultInfo`](/rules/lib/providers/DefaultInfo):

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
*predeclared outputs* (generally, those created by [output
attributes](#output_attributes)).

Rules that perform actions should provide default outputs, even if those outputs
are not expected to be directly used. Actions that are not in the graph of the
requested outputs are pruned. If an output is only used by a target's consumers,
those actions will not be performed when the target is built in isolation. This
makes debugging more difficult because rebuilding just the failing target won't
reproduce the failure.

#### Runfiles

Runfiles are a set of files used by a target at runtime (as opposed to build
time). During the [execution phase](/extending/concepts#evaluation-model), Bazel creates
a directory tree containing symlinks pointing to the runfiles. This stages the
environment for the binary so it can access the runfiles during runtime.

Runfiles can be added manually during rule creation.
[`runfiles`](/rules/lib/builtins/runfiles) objects can be created by the `runfiles` method
on the rule context, [`ctx.runfiles`](/rules/lib/builtins/ctx#runfiles) and passed to the
`runfiles` parameter on `DefaultInfo`. The executable output of
[executable rules](#executable-rules) is implicitly added to the runfiles.

Some rules specify attributes, generally named
[`data`](/reference/be/common-definitions#common.data), whose outputs are added to
a targets' runfiles. Runfiles should also be merged in from `data`, as well as
from any attributes which might provide code for eventual execution, generally
`srcs` (which might contain `filegroup` targets with associated `data`) and
`deps`.

```python
def _example_library_impl(ctx):
    ...
    runfiles = ctx.runfiles(files = ctx.files.data)
    transitive_runfiles = []
    for runfiles_attr in (
        ctx.attr.srcs,
        ctx.attr.hdrs,
        ctx.attr.deps,
        ctx.attr.data,
    ):
        for target in runfiles_attr:
            transitive_runfiles.append(target[DefaultInfo].default_runfiles)
    runfiles = runfiles.merge_all(transitive_runfiles)
    return [
        DefaultInfo(..., runfiles = runfiles),
        ...
    ]
```

#### Custom providers

Providers can be defined using the [`provider`](/rules/lib/globals/bzl#provider)
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

##### Custom initialization of providers

It's possible to guard the instantiation of a provider with custom
preprocessing and validation logic. This can be used to ensure that all
provider instances obey certain invariants, or to give users a cleaner API for
obtaining an instance.

This is done by passing an `init` callback to the
[`provider`](/rules/lib/globals/bzl.html#provider) function. If this callback is given, the
return type of `provider()` changes to be a tuple of two values: the provider
symbol that is the ordinary return value when `init` is not used, and a "raw
constructor".

In this case, when the provider symbol is called, instead of directly returning
a new instance, it will forward the arguments along to the `init` callback. The
callback's return value must be a dict mapping field names (strings) to values;
this is used to initialize the fields of the new instance. Note that the
callback may have any signature, and if the arguments do not match the signature
an error is reported as if the callback were invoked directly.

The raw constructor, by contrast, will bypass the `init` callback.

The following example uses `init` to preprocess and validate its arguments:

```python
# //pkg:exampleinfo.bzl

_core_headers = [...]  # private constant representing standard library files

# It's possible to define an init accepting positional arguments, but
# keyword-only arguments are preferred.
def _exampleinfo_init(*, files_to_link, headers = None, allow_empty_files_to_link = False):
    if not files_to_link and not allow_empty_files_to_link:
        fail("files_to_link may not be empty")
    all_headers = depset(_core_headers, transitive = headers)
    return {'files_to_link': files_to_link, 'headers': all_headers}

ExampleInfo, _new_exampleinfo = provider(
    ...
    init = _exampleinfo_init)

export ExampleInfo
```

A rule implementation may then instantiate the provider as follows:

```python
    ExampleInfo(
        files_to_link=my_files_to_link,  # may not be empty
        headers = my_headers,  # will automatically include the core headers
    )
```

The raw constructor can be used to define alternative public factory functions
that do not go through the `init` logic. For example, in exampleinfo.bzl we
could define:

```python
def make_barebones_exampleinfo(headers):
    """Returns an ExampleInfo with no files_to_link and only the specified headers."""
    return _new_exampleinfo(files_to_link = depset(), headers = all_headers)
```

Typically, the raw constructor is bound to a variable whose name begins with an
underscore (`_new_exampleinfo` above), so that user code cannot load it and
generate arbitrary provider instances.

Another use for `init` is to simply prevent the user from calling the provider
symbol altogether, and force them to use a factory function instead:

```python
def _exampleinfo_init_banned(*args, **kwargs):
    fail("Do not call ExampleInfo(). Use make_exampleinfo() instead.")

ExampleInfo, _new_exampleinfo = provider(
    ...
    init = _exampleinfo_init_banned)

def make_exampleinfo(...):
    ...
    return _new_exampleinfo(...)
```

<a name="executable-rules"></a>

## Executable rules and test rules

Executable rules define targets that can be invoked by a `bazel run` command.
Test rules are a special kind of executable rule whose targets can also be
invoked by a `bazel test` command. Executable and test rules are created by
setting the respective [`executable`](/rules/lib/globals/bzl#rule.executable) or
[`test`](/rules/lib/globals/bzl#rule.test) argument to `True` in the call to `rule`:

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
`executable` argument of a returned [`DefaultInfo`](/rules/lib/providers/DefaultInfo)
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
a [`ctx.actions.run`](/rules/lib/builtins/actions#run) or
[`ctx.actions.run_shell`](/rules/lib/builtins/actions#run_shell) action this should be done
by the underlying tool that is invoked by the action. For a
[`ctx.actions.write`](/rules/lib/builtins/actions#write) action, pass `is_executable=True`.

As [legacy behavior](#deprecated_predeclared_outputs), executable rules have a
special `ctx.outputs.executable` predeclared output. This file serves as the
default executable if you do not specify one using `DefaultInfo`; it must not be
used otherwise. This output mechanism is deprecated because it does not support
customizing the executable file's name at analysis time.

See examples of an
[executable rule](https://github.com/bazelbuild/examples/blob/main/rules/executable/fortune.bzl){: .external}
and a
[test rule](https://github.com/bazelbuild/examples/blob/main/rules/test_rule/line_length.bzl){: .external}.

[Executable rules](/reference/be/common-definitions#common-attributes-binaries) and
[test rules](/reference/be/common-definitions#common-attributes-tests) have additional
attributes implicitly defined, in addition to those added for
[all rules](/reference/be/common-definitions#common-attributes). The defaults of
implicitly-added attributes cannot be changed, though this can be worked around
by wrapping a private rule in a [Starlark macro](/extending/macros) which alters the
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
# Given launcher_path and runfile_file:
runfiles_root = launcher_path.path + ".runfiles"
workspace_name = ctx.workspace_name
runfile_path = runfile_file.short_path
execution_root_relative_path = "%s/%s/%s" % (
    runfiles_root, workspace_name, runfile_path)
```

The path to a `File` under the runfiles directory corresponds to
[`File.short_path`](/rules/lib/builtins/File#short_path).

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

In addition to [default outputs](#default_outputs), any *predeclared output* can
be explicitly requested on the command line. Rules can specify predeclared
outputs via [output attributes](#output_attributes). In that case, the user
explicitly chooses labels for outputs when they instantiate the rule. To obtain
[`File`](/rules/lib/builtins/File) objects for output attributes, use the corresponding
attribute of [`ctx.outputs`](/rules/lib/builtins/ctx#outputs). Rules can
[implicitly define predeclared outputs](#deprecated_predeclared_outputs) based
on the target name as well, but this feature is deprecated.

In addition to default outputs, there are *output groups*, which are collections
of output files that may be requested together. These can be requested with
[`--output_groups`](/reference/command-line-reference#flag--output_groups). For
example, if a target `//pkg:mytarget` is of a rule type that has a `debug_files`
output group, these files can be built by running `bazel build //pkg:mytarget
--output_groups=debug_files`. Since non-predeclared outputs don't have labels,
they can only be requested by appearing in the default outputs or an output
group.

Output groups can be specified with the
[`OutputGroupInfo`](/rules/lib/providers/OutputGroupInfo) provider. Note that unlike many
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

Also unlike most providers, `OutputGroupInfo` can be returned by both an
[aspect](/extending/aspects) and the rule target to which that aspect is applied, as
long as they do not define the same output groups. In that case, the resulting
providers are merged.

Note that `OutputGroupInfo` generally shouldn't be used to convey specific sorts
of files from a target to the actions of its consumers. Define
[rule-specific providers](#custom_providers) for that instead.

### Configurations

Imagine that you want to build a C++ binary for a different architecture. The
build can be complex and involve multiple steps. Some of the intermediate
binaries, like compilers and code generators, have to run on
[the execution platform](/extending/platforms#overview) (which could be your host,
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
[See example](https://github.com/bazelbuild/examples/blob/main/rules/actions_run/execute.bzl){: .external}

In general, sources, dependent libraries, and executables that will be needed at
runtime can use the same configuration.

Tools that are executed as part of the build (such as compilers or code generators)
should be built for an exec configuration. In this case, specify `cfg="exec"` in
the attribute.

Otherwise, executables that are used at runtime (such as as part of a test) should
be built for the target configuration. In this case, specify `cfg="target"` in
the attribute.

`cfg="target"` doesn't actually do anything: it's purely a convenience value to
help rule designers be explicit about their intentions. When `executable=False`,
which means `cfg` is optional, only set this when it truly helps readability.

You can also use `cfg=my_transition` to use
[user-defined transitions](/extending/config#user-defined-transitions), which allow
rule authors a great deal of flexibility in changing configurations, with the
drawback of
[making the build graph larger and less comprehensible](/extending/config#memory-and-performance-considerations).

**Note**: Historically, Bazel didn't have the concept of execution platforms,
and instead all build actions were considered to run on the host machine. Bazel
versions before 6.0 created a distinct "host" configuration to represent this.
If you see references to "host" in code or old documentation, that's what this
refers to. We recommend using Bazel 6.0 or newer to avoid this extra conceptual
overhead.

<a name="fragments"></a>

### Configuration fragments

Rules may access
[configuration fragments](/rules/lib/fragments) such as
`cpp`, `java` and `jvm`. However, all required fragments must be declared in
order to avoid access errors:

```python
def _impl(ctx):
    # Using ctx.fragments.cpp leads to an error since it was not declared.
    x = ctx.fragments.java
    ...

my_rule = rule(
    implementation = _impl,
    fragments = ["java"],      # Required fragments of the target configuration
    host_fragments = ["java"], # Required fragments of the host configuration
    ...
)
```

### Runfiles symlinks

Normally, the relative path of a file in the runfiles tree is the same as the
relative path of that file in the source tree or generated output tree. If these
need to be different for some reason, you can specify the `root_symlinks` or
`symlinks` arguments. The `root_symlinks` is a dictionary mapping paths to
files, where the paths are relative to the root of the runfiles directory. The
`symlinks` dictionary is the same, but paths are implicitly prefixed with the
name of the main workspace (*not* the name of the repository containing the
current target).

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

### Code coverage

When the [`coverage`](/reference/command-line-reference#coverage) command is run,
the build may need to add coverage instrumentation for certain targets. The
build also gathers the list of source files that are instrumented. The subset of
targets that are considered is controlled by the flag
[`--instrumentation_filter`](/reference/command-line-reference#flag--instrumentation_filter).
Test targets are excluded, unless
[`--instrument_test_targets`](/reference/command-line-reference#flag--instrument_test_targets)
is specified.

If a rule implementation adds coverage instrumentation at build time, it needs
to account for that in its implementation function.
[ctx.coverage_instrumented](/rules/lib/builtins/ctx#coverage_instrumented) returns true in
coverage mode if a target's sources should be instrumented:

```python
# Are this rule's sources instrumented?
if ctx.coverage_instrumented():
  # Do something to turn on coverage for this compile action
```

Logic that always needs to be on in coverage mode (whether a target's sources
specifically are instrumented or not) can be conditioned on
[ctx.configuration.coverage_enabled](/rules/lib/builtins/configuration#coverage_enabled).

If the rule directly includes sources from its dependencies before compilation
(such as header files), it may also need to turn on compile-time instrumentation if
the dependencies' sources should be instrumented:

```python
# Are this rule's sources or any of the sources for its direct dependencies
# in deps instrumented?
if (ctx.configuration.coverage_enabled and
    (ctx.coverage_instrumented() or
     any([ctx.coverage_instrumented(dep) for dep in ctx.attr.deps]))):
    # Do something to turn on coverage for this compile action
```

Rules also should provide information about which attributes are relevant for
coverage with the `InstrumentedFilesInfo` provider, constructed using
[`coverage_common.instrumented_files_info`](/rules/lib/toplevel/coverage_common#instrumented_files_info).
The `dependency_attributes` parameter of `instrumented_files_info` should list
all runtime dependency attributes, including code dependencies like `deps` and
data dependencies like `data`. The `source_attributes` parameter should list the
rule's source files attributes if coverage instrumentation might be added:

```python
def _example_library_impl(ctx):
    ...
    return [
        ...
        coverage_common.instrumented_files_info(
            ctx,
            dependency_attributes = ["deps", "data"],
            # Omitted if coverage is not supported for this rule:
            source_attributes = ["srcs", "hdrs"],
        )
        ...
    ]
```

If `InstrumentedFilesInfo` is not returned, a default one is created with each
non-tool [dependency attribute](#dependency_attributes) that doesn't set
[`cfg`](#configuration) to `"host"` or `"exec"` in the attribute schema) in
`dependency_attributes`. (This isn't ideal behavior, since it puts attributes
like `srcs` in `dependency_attributes` instead of `source_attributes`, but it
avoids the need for explicit coverage configuration for all rules in the
dependency chain.)

### Validation Actions

Sometimes you need to validate something about the build, and the
information required to do that validation is available only in artifacts
(source files or generated files). Because this information is in artifacts,
rules cannot do this validation at analysis time because rules cannot read
files. Instead, actions must do this validation at execution time. When
validation fails, the action will fail, and hence so will the build.

Examples of validations that might be run are static analysis, linting,
dependency and consistency checks, and style checks.

Validation actions can also help to improve build performance by moving parts
of actions that are not required for building artifacts into separate actions.
For example, if a single action that does compilation and linting can be
separated into a compilation action and a linting action, then the linting
action can be run as a validation action and run in parallel with other actions.

These "validation actions" often don't produce anything that is used elsewhere
in the build, since they only need to assert things about their inputs. This
presents a problem though: If a validation action does not produce anything that
is used elsewhere in the build, how does a rule get the action to run?
Historically, the approach was to have the validation action output an empty
file, and artificially add that output to the inputs of some other important
action in the build:

<img src="/rules/validation_action_historical.svg" width="35%" />

This works, because Bazel will always run the validation action when the compile
action is run, but this has significant drawbacks:

1. The validation action is in the critical path of the build. Because Bazel
thinks the empty output is required to run the compile action, it will run the
validation action first, even though the compile action will ignore the input.
This reduces parallelism and slows down builds.

2. If other actions in the build might run instead of the
compile action, then the empty outputs of validation actions need to be added to
those actions as well (`java_library`'s source jar output, for example). This is
also a problem if new actions that might run instead of the compile action are
added later, and the empty validation output is accidentally left off.

The solution to these problems is to use the Validations Output Group.

#### Validations Output Group

The Validations Output Group is an output group designed to hold the otherwise
unused outputs of validation actions, so that they don't need to be artificially
added to the inputs of other actions.

This group is special in that its outputs are always requested, regardless of
the value of the `--output_groups` flag, and regardless of how the target is
depended upon (for example, on the command line, as a dependency, or through
implicit outputs of the target). Note that normal caching and incrementality
still apply: if the inputs to the validation action have not changed and the
validation action previously succeeded, then the validation action will not be
run.

<img src="/rules/validation_action.svg" width="35%" />

Using this output group still requires that validation actions output some file,
even an empty one. This might require wrapping some tools that normally don't
create outputs so that a file is created.

A target's validation actions are not run in three cases:

*    When the target is depended upon as a tool
*    When the target is depended upon as an implicit dependency (for example, an
     attribute that starts with "_")
*    When the target is built in the host or exec configuration.

It is assumed that these targets have their own
separate builds and tests that would uncover any validation failures.

#### Using the Validations Output Group

The Validations Output Group is named `_validation` and is used like any other
output group:

```python
def _rule_with_validation_impl(ctx):

  ctx.actions.write(ctx.outputs.main, "main output\n")

  ctx.actions.write(ctx.outputs.implicit, "implicit output\n")

  validation_output = ctx.actions.declare_file(ctx.attr.name + ".validation")
  ctx.actions.run(
      outputs = [validation_output],
      executable = ctx.executable._validation_tool,
      arguments = [validation_output.path])

  return [
    DefaultInfo(files = depset([ctx.outputs.main])),
    OutputGroupInfo(_validation = depset([validation_output])),
  ]


rule_with_validation = rule(
  implementation = _rule_with_validation_impl,
  outputs = {
    "main": "%{name}.main",
    "implicit": "%{name}.implicit",
  },
  attrs = {
    "_validation_tool": attr.label(
        default = Label("//validation_actions:validation_tool"),
        executable = True,
        cfg = "exec"),
  }
)
```

Notice that the validation output file is not added to the `DefaultInfo` or the
inputs to any other action. The validation action for a target of this rule kind
will still run if the target is depended upon by label, or any of the target's
implicit outputs are directly or indirectly depended upon.

It is usually important that the outputs of validation actions only go into the
validation output group, and are not added to the inputs of other actions, as
this could defeat parallelism gains. Note however that Bazel does not currently
have any special checking to enforce this. Therefore, you should test
that validation action outputs are not added to the inputs of any actions in the
tests for Starlark rules. For example:

```python
load("@bazel_skylib//lib:unittest.bzl", "analysistest")

def _validation_outputs_test_impl(ctx):
  env = analysistest.begin(ctx)

  actions = analysistest.target_actions(env)
  target = analysistest.target_under_test(env)
  validation_outputs = target.output_groups._validation.to_list()
  for action in actions:
    for validation_output in validation_outputs:
      if validation_output in action.inputs.to_list():
        analysistest.fail(env,
            "%s is a validation action output, but is an input to action %s" % (
                validation_output, action))

  return analysistest.end(env)

validation_outputs_test = analysistest.make(_validation_outputs_test_impl)
```

#### Validation Actions Flag

Running validation actions is controlled by the `--run_validations` command line
flag, which defaults to true.

## Deprecated features

### Deprecated predeclared outputs

There are two **deprecated** ways of using predeclared outputs:

*   The [`outputs`](/rules/lib/globals/bzl#rule.outputs) parameter of `rule` specifies
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

[`ctx.runfiles`](/rules/lib/builtins/ctx#runfiles) and the [`runfiles`](/rules/lib/builtins/runfiles)
type have a complex set of features, many of which are kept for legacy reasons.
The following recommendations help reduce complexity:

*   **Avoid** use of the `collect_data` and `collect_default` modes of
    [`ctx.runfiles`](/rules/lib/builtins/ctx#runfiles). These modes implicitly collect
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
    [`data`](/reference/be/common-definitions#common-attributes.data)).

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
    [`DefaultInfo`](/rules/lib/providers/DefaultInfo). It is not allowed to specify any of
    these fields while also returning a `DefaultInfo` provider.

*   The field `output_groups` takes a struct value and corresponds to an
    [`OutputGroupInfo`](/rules/lib/providers/OutputGroupInfo).

In [`provides`](/rules/lib/globals/bzl#rule.provides) declarations of rules, and in
[`providers`](/rules/lib/toplevel/attr#label_list.providers) declarations of dependency
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

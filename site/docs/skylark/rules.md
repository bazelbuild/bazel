---
layout: documentation
title: Rules
---

# Rules

A rule defines a series of [actions](#actions) that Bazel should perform on
inputs to get a set of outputs. For example, a C++ binary rule might take a set
of `.cpp` files (the inputs), run `g++` on them (the action), and return an
executable file (the output).

Note that, from Bazel's perspective, `g++` and the standard C++ libraries are
also inputs to this rule. As a rule writer, you must consider not only the
user-provided inputs to a rule, but also all of the tools and libraries required
to execute the actions.

Before creating or modifying any rule, make sure you are familiar with the
[evaluation model](concepts.md). You must understand the three phases of
execution and the differences between macros and rules.

A few rules are built into Bazel itself. These *native rules*, such as
`cc_library` and `java_binary`, provide some core support for certain languages.
By defining your own rules, you can add similar support for languages and tools
that Bazel does not support natively.

Rules defined in .bzl files work just like native rules. For example, their
targets have labels, can appear in `bazel query`, and get built whenever they
are needed for a `bazel build` command or similar. When defining your own rule,
you get to decide what attributes it supports and how it generates its outputs.

The exact behavior of a rule during the
[analysis phase](concepts.md#evaluation-model) is governed by its
`implementation` function. This function does not run any external commands.
Rather, it registers [actions](#actions) that will be used later during the
execution phase to build the rule's outputs, if they are needed. Rules also
produce and pass along information that may be useful to other rules, in the
form of [providers](#providers).

## Contents

*  [Rule creation](#rule-creation)
*  [Attributes](#attributes)
*  [Private Attributes and Implicit Dependencies](#private-attributes-and-implicit-dependencies)
*  [Implementation function](#implementation-function)
*  [Targets](#targets)
*  [Files](#files)
*  [Actions](#actions)
*  [Configurations](#configurations)
*  [Configuration Fragments](#configuration-fragments)
*  [Providers](#providers)
*  [Runfiles](#runfiles)
*  [Requesting output files](#requesting-output-files)
*  [Code coverage instrumentation](#code-coverage-instrumentation)
*  [Executable rules and test rules](#executable-rules-and-test-rules)


## Rule creation

In a `.bzl` file, use the [rule](lib/globals.html#rule)
function to create a new rule and store it in a global variable:

```python
my_rule = rule(...)
```

The rule can then be loaded in `BUILD` files:

```python
load('//some/pkg:whatever.bzl', 'my_rule')
```

[See example](https://github.com/bazelbuild/examples/tree/master/rules/empty).

## Attributes

An attribute is a rule argument, such as `srcs` or `deps`. You must list the
names and schemas of all attributes when you define a rule. Attribute schemas
are created using the [attr](lib/attr.html) module.

```python
sum = rule(
    implementation = _impl,
    attrs = {
        "number": attr.int(default = 1),
        "deps": attr.label_list(),
    },
)
```

In a `BUILD` file, call the rule to create targets of this type:

```python
sum(
    name = "my-target",
    deps = [":other-target"],
)

sum(
    name = "other-target",
)
```

Here `other-target` is a dependency of `my-target`, and therefore `other-target`
will be analyzed first.

There are two special kinds of attributes:

* *Dependency attributes*, such as `attr.label` and `attr.label_list`,
  declare a dependency from the target that owns the attribute to the target
  whose label appears in the attribute's value. This kind of attribute forms the
  basis of the target graph.

* *Output attributes*, such as `attr.output` and `attr.output_list`, declare an
  output file that the target generates. Although they refer to the output file
  by label, they do not create a dependency relationship between targets. Output
  attributes are used relatively rarely, in favor of other ways of declaring
  output files that do not require the user to specify a label.

Both dependency attributes and output attributes take in label values. These may
be specified as either [`Label`](lib/Label.html) objects or as simple strings.
If a string is given, it will be converted to a `Label` using the
[constructor](lib/Label.html#Label). The repository, and possibly the path, will
be resolved relative to the defined target.

If an attribute schema is defined in the rule but no value for that attribute is
given when the rule is instantiated, then the rule implementation function will
see a placeholder value in `ctx.attr`. The placeholder value depends on the type
of attribute. If the schema specifies a `default` value, that value will be used
instead of the placeholder. The schema may also specify `mandatory=True`, in
which case it is illegal for the user to not give an explicit value. It is not
useful for an attribute schema with `mandatory` to also have a `default`.

The following attributes are automatically added to every rule: `deprecation`,
`features`, `name`, `tags`, `testonly`, `visibility`. Test rules also have the
following attributes: `args`, `flaky`, `local`, `shard_count`, `size`,
`timeout`.

### <a name="private-attributes"></a> Private Attributes and Implicit Dependencies

A dependency attribute with a default value is called an *implicit dependency*.
The name comes from the fact that it is a part of the target graph that the user
does not specify in a BUILD file. Implicit dependencies are useful for
hard-coding a relationship between a rule and a tool (such as a compiler), since
most of the time a user is not interested in specifying what tool the rule uses.
From the rule's point of view, the tool is still an input, just like any source
file or other dependency.

Sometimes we want to not only provide a default value, but prevent the user from
overriding this default. To do this, you can make the attribute *private* by
giving it a name that begins with an underscore (`_`). Private attributes must
have default values. It generally only makes sense to use private attributes for
implicit dependencies.

```python
metal_binary = rule(
    implementation = _metal_binary_impl,
    attrs = {
        "srcs": attr.label_list(),
        "_compiler": attr.label(
            default = Label("//tools:metalc"),
            allow_single_file = True,
            executable = True,
        ),
    },
)
```

In this example, every target of type `metal_binary` will have an implicit
dependency on the compiler `//tools:metalc`. This allows `metal_binary`'s
implementation function to generate actions that invoke the compiler, even
though the user did not pass its label as an input. Since `_compiler` is a
private attribute, we know for sure that `ctx.attr._compiler` will always point
to `//tools:metalc` in all targets of this rule type. Alternatively, we could
have named the attribute `compiler` without the underscore and kept the default
value. This lets users substitute a different compiler if necessary, but
requires no awareness of the compiler's label otherwise.

## Implementation function

Every rule requires an `implementation` function. This function contains the
actual logic of the rule and is executed strictly in the
[analysis phase](concepts.md#evaluation-model). As such, the function is not
able to actually read or write files. Rather, its main job is to emit
[actions](#actions) that will run later during the execution phase.

Implementation functions take exactly one parameter: a [rule
context](lib/ctx.html), conventionally named `ctx`. It can be used to:

* access attribute values and obtain handles on declared input and output files;

* create actions; and

* pass information to other targets that depend on this one, via
  [providers](#providers).

The most common way to access attribute values is by using
`ctx.attr.<attribute_name>`, though there are several other fields besides
`attr` that provide more convenient ways of accessing file handles, such as
`ctx.file` and `ctx.outputs`. The name and the package of a rule are available
with `ctx.label.name` and `ctx.label.package`. The `ctx` object also contains
some helper functions. See its [documentation](lib/ctx.html) for a complete
list.

Rule implementation functions are usually private (i.e., named with a leading
underscore) because they tend not to be reused. Conventionally, they are named
the same as their rule, but suffixed with `_impl`.

See [an example](https://github.com/bazelbuild/examples/blob/master/rules/attributes/printer.bzl)
of declaring and accessing attributes.

## Targets

Each call to a build rule returns no value but has the side effect of defining a
new target; this is called instantiating the rule. The dependencies of the new
target are any other targets whose labels are mentioned in its dependency
attributes. In the following example, the target `//mypkg:y` depends on the
targets `//mypkg:x` and `//mypkg:z.foo`.

```python
# //mypkg:BUILD

my_rule(
    name = "x",
)

# Assuming that my_rule has attributes "deps" and "srcs",
# of type attr.label_list()
my_rule(
    name = "y",
    deps = [":x"],
    srcs = [":z.foo"],
)
```

Dependencies are represented at analysis time as [`Target`](lib/Target.html)
objects. These objects contain the information produced by analyzing a target --
in particular, its [providers](#providers). The current target can access its
dependencies' `Target` objects within its rule implementation function by using
`ctx.attr`.

## Files

Files are represented by the [`File`](lib/File.html) type. Since Bazel does not
perform file I/O during the analysis phase, these objects cannot be used to
directly read or write file content. Rather, they are passed to action-emitting
functions to construct pieces of the action graph. See
[`ctx.actions`](lib/actions.html) for the available kinds of actions.

A file can either be a source file or a generated file. Each generated file must
be an output of exactly one action. Source files cannot be the output of any
action.

Some files, including all source files, are addressable by labels. These files
have `Target` objects associated with them. If a file's label appears within a
dependency attribute (for example, in a `srcs` attribute of type
`attr.label_list`), the `ctx.attr.<attr_name>` entry for it will contain the
corresponding `Target`. The `File` object can be obtained from this `Target`'s
`files` field. This allows the file to be referenced in both the target graph
and the action graph.

During the analysis phase, a rule's implementation function can create
additional output files. Since all labels have to be known during the loading
phase, these additional output files are not associated with labels or
`Target`s. Generally these are intermediate files needed for a later compilation
step, or auxiliary outputs that don't need to be referenced in the target graph.
Even though these files don't have a label, they can still be passed along in a
[`provider`](#providers) to make them available to other depending targets at
analysis time.

A generated file that is addressable by a label is called a *predeclared
output*. There are multiple ways for a rule to introduce a predeclared output:

* If the rule declares an [`outputs`](lib/globals.html#rule.outputs) dict in its
  call to `rule()`, then each entry in this dict becomes an output. The output's
  label is chosen automatically as specified by the entry, usually by
  substituting into a string template. This is the most common way to define
  outputs.

* The rule can have an attribute of type [`output`](lib/attr.html#output) or
  [`output_list`](lib/attr.html#output_list). In this case the user explicitly
  chooses the label for the output when they instantiate the rule.

* **(Deprecated)** If the rule is marked
  [`executable`](lib/globals.html#rule.executable) or
  [`test`](lib/globals.html#rule.test), an output is created with the same name
  as the rule instance itself. (Technically, the file has no label since it
  would clash with the rule instance's own label, but it is still considered a
  predeclared output.) By default, this file serves as the binary to run if the
  target appears on the command line of a `bazel run` or `bazel test` command.
  See [Executable rules](#executable-rules-and-test-rules) below.

All predeclared outputs can be accessed within the rule's implementation
function under the [`ctx.outputs`](lib/ctx.html#outputs) struct; see that page
for details and restrictions. Non-predeclared outputs are created during
analysis using the [`ctx.actions.declare_file`](lib/actions.html#declare_file)
and [`ctx.actions.declare_directory`](lib/actions.html#declare_directory)
functions. Both kinds of outputs may be passed along in providers.

Although the input files of a target -- those files passed through dependency
attributes -- can be accessed indirectly via `ctx.attr`, it is more convenient
to use `ctx.file` and `ctx.files`. For output files that are predeclared using
output attributes (attributes of type `attr.output` or `attr.output_list`),
`ctx.attr` will only return the label, and you must use `ctx.outputs` to get the
actual `File` object.

[See example of predeclared outputs](https://github.com/bazelbuild/examples/blob/master/rules/predeclared_outputs/hash.bzl)

## Actions

An action describes how to generate a set of outputs from a set of inputs, for
example "run gcc on hello.c and get hello.o". When an action is created, Bazel
doesn't run the command immediately. It registers it in a graph of dependencies,
because an action can depend on the output of another action (e.g. in C,
the linker must be called after compilation). In the execution phase, Bazel
decides which actions must be run and in which order.

All functions that create actions are defined in [`ctx.actions`](lib/actions.html):

* [ctx.actions.run](lib/actions.html#run), to run an executable.
* [ctx.actions.run_shell](lib/actions.html#run_shell), to run a shell command.
* [ctx.actions.write](lib/actions.html#write), to write a string to a file.
* [ctx.actions.expand_template](lib/actions.html#expand_template), to generate a file from a template.

Actions take a set (which can be empty) of input files and generate a (non-empty)
set of output files.
The set of input and output files must be known during the
[analysis phase](concepts.md#evaluation-model). It might depend on the value
of attributes and information from dependencies, but it cannot depend on the
result of the execution. For example, if your action runs the unzip command, you
must specify which files you expect to be inflated (before running unzip).

Actions are comparable to pure functions: They should depend only on the
provided inputs, and avoid accessing computer information, username, clock,
network, or I/O devices (except for reading inputs and writing outputs). This is
important because the output will be cached and reused.

**If an action generates a file that is not listed in its outputs**: This is
fine, but the file will be ignored and cannot be used by other rules.

**If an action does not generate a file that is listed in its outputs**: This is
an execution error and the build will fail. This happens for instance when a
compilation fails.

**If an action generates an unknown number of outputs and you want to keep them
all**, you must group them in a single file (e.g., a zip, tar, or other
archive format). This way, you will be able to deterministically declare your
outputs.

**If an action does not list a file it uses as an input**, the action execution
will most likely result in an error. The file is not guaranteed to be available
to the action, so if it **is** there, it's due to coincidence or error.

**If an action lists a file as an input, but does not use it**: This is fine.
However, it can affect action execution order, resulting in sub-optimal
performance.

Dependencies are resolved by Bazel, which will decide which actions are
executed. It is an error if there is a cycle in the dependency graph. Creating
an action does not guarantee that it will be executed: It depends on whether
its outputs are needed for the build.

## Configurations

Imagine that you want to build a C++ binary and target a different architecture.
The build can be complex and involve multiple steps. Some of the intermediate
binaries, like the compilers and code generators, have to run on your machine
(the host); some of the binaries such the final output must be built for the
target architecture.

For this reason, Bazel has a concept of "configurations" and transitions. The
topmost targets (the ones requested on the command line) are built in the
"target" configuration, while tools that should run locally on the host are
built in the "host" configuration. Rules may generate different actions based on
the configuration, for instance to change the cpu architecture that is passed to
the compiler. In some cases, the same library may be needed for different
configurations. If this happens, it will be analyzed and potentially built
multiple times.

By default, Bazel builds the dependencies of a target in the same configuration
as the target itself, i.e. without transitioning. When a target depends on a
tool, the label attribute will specify a transition to the host configuration.
This causes the tool and all of its dependencies to be built for the host
machine, assuming those dependencies do not themselves have transitions.

For each dependency attribute, you can decide whether the dependency target
should be built in the same configuration, or transition to the host
configuration (using `cfg`). If a dependency attribute has the flag
`executable=True`, the configuration must be set explicitly.
[See example](https://github.com/bazelbuild/examples/blob/master/rules/actions_run/execute.bzl)

In general, sources, dependent libraries, and executables that will be needed at
runtime can use the same configuration.

Tools that are executed as part of the build (e.g., compilers, code generators)
should be built for the host configuration. In this case, specify `cfg="host"`
in the attribute.

Otherwise, executables that are used at runtime (e.g. as part of a test) should
be built for the target configuration. In this case, specify `cfg="target"` in
the attribute.

## <a name="fragments"></a> Configuration Fragments

Rules may access [configuration fragments](lib/skylark-configuration-fragment.html)
such as `cpp`, `java` and `jvm`. However, all required fragments must be
declared in order to avoid access errors:

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
configuration. If you want to access fragments for the host configuration,
use `ctx.host_fragments` instead.

## Providers

Providers are pieces of information that a rule exposes to other rules that
depend on it. This data can include output files, libraries, parameters to pass
on a tool's command line, or anything else the depending rule should know about.
Providers are the only mechanism to exchange data between rules, and can be
thought of as part of a rule's public interface (loosely analogous to a
function's return value).

A rule can only see the providers of its direct dependencies. If there is a rule
`top` that depends on `middle`, and `middle` depends on `bottom`, then we say
that `middle` is a direct dependency of `top`, while `bottom` is a transitive
dependency of `top`. In this case, `top` can see the providers of `middle`. The
only way for `top` to see any information from `bottom` is if `middle`
re-exports this information in its own providers; this is how transitive
information can be accumulated from all dependencies. In such cases, consider
using [depsets](depsets.md) to hold the data more efficiently without excessive
copying.

Providers can be declared using the [provider()](lib/globals.html#provider) function:

```python
TransitiveDataInfo = provider(fields=["value"])
```

Rule implementation function can then construct and return provider instances:

```python
def rule_implementation(ctx):
  ...
  return [TransitiveDataInfo(value=5)]
```

`TransitiveDataInfo` acts both as a constructor for provider instances and as a key to access them.
A [target](lib/Target.html) serves as a map from each provider that the target supports, to the
target's corresponding instance of that provider.
A rule can access the providers of its dependencies using the square bracket notation (`[]`):

```python
def dependent_rule_implementation(ctx):
  ...
  n = 0
  for dep_target in ctx.attr.deps:
    n += dep_target[TransitiveDataInfo].value
  ...
```

All targets have a [`DefaultInfo`](lib/DefaultInfo.html) provider that can be used to access
some information relevant to all targets.

Providers are only available during the analysis phase. Examples of usage:

* [mandatory providers](https://github.com/bazelbuild/examples/blob/master/rules/mandatory_provider/sum.bzl)
* [optional providers](https://github.com/bazelbuild/examples/blob/master/rules/optional_provider/sum.bzl)
* [providers with depsets](https://github.com/bazelbuild/examples/blob/master/rules/depsets/foo.bzl)
    This examples shows how a library and a binary rule can pass information.

### Migrating from Legacy Providers

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
def _myrule_impl(ctx):
  ...
  legacy_data = struct(x="foo", ...)
  modern_data = MyInfo(y="bar", ...)
  # When any legacy providers are returned, the top-level returned value is a struct.
  return struct(
      # One key = value entry for each legacy provider.
      legacy_info = legacy_data,
      ...
      # All modern providers are put in a list passed to the special "providers" key.
      providers = [modern_data, ...])
```

If `dep` is the resulting `Target` object for an instance of this rule, the
providers and their contents can be retrieved as `dep.legacy_info.x` and
`dep[MyInfo].y`.

In addition to `providers`, the returned struct can also take several other
fields that have special meaning (and that do not create a corresponding legacy
provider).

* The fields `files`, `runfiles`, `data_runfiles`, `default_runfiles`, and
  `executable` correspond to the same-named fields of
  [`DefaultInfo`](lib/DefaultInfo.html). It is not allowed to specify
  any of these fields while also returning a `DefaultInfo` modern provider.

* The field `output_groups` takes a struct value and corresponds to an
  [`OutputGroupInfo`](lib/OutputGroupInfo.html).

* The field `instrumented_files` is for
  [code coverage instrumentation](#code-coverage-instrumentation). It does not
  yet have a modern provider equivalent. If you need it, you cannot yet migrate
  away from legacy providers.

In [`provides`](lib/globals.html#rule.provides) declarations of rules, and in
[`providers`](lib/attr.html#label_list.providers) declarations of dependency
attributes, legacy providers are passed in as strings and modern providers are
passed in by their `*Info` symbol. Be sure to change from strings to symbols
when migrating. For complex or large rule sets where it is difficult to update
all rules atomically, you may have an easier time if you follow this sequence of
steps:

1. Modify the rules that produce the legacy provider to produce both the legacy
   and modern providers, using the above syntax. For rules that declare they
   return the legacy provider, update that declaration to include both the
   legacy and modern providers.

2. Modify the rules that consume the legacy provider to instead consume the
   modern provider. If any attribute declarations require the legacy provider,
   also update them to instead require the modern provider. Optionally, you can
   interleave this work with step 1 by having consumers accept/require either
   provider: Test for the presence of the legacy provider using
   `hasattr(target, 'foo')`, or the new provider using `FooInfo in target`.

3. Fully remove the legacy provider from all rules.

## Runfiles

Runfiles are a set of files used by the (often executable) output of a rule
during runtime (as opposed to build time, i.e. when the binary itself is
generated).
During the [execution phase](concepts.md#evaluation-model), Bazel creates a
directory tree containing symlinks pointing to the runfiles. This stages the
environment for the binary so it can access the runfiles during runtime.

[See example](https://github.com/bazelbuild/examples/blob/master/rules/runfiles/execute.bzl)

Runfiles can be added manually during rule creation and/or collected
transitively from the rule's dependencies:

```python
def _rule_implementation(ctx):
  ...
  transitive_runfiles = depset(transitive=
    [dep.transitive_runtime_files for dep in ctx.attr.special_dependencies])

  runfiles = ctx.runfiles(
      # Add some files manually.
      files = [ctx.file.some_data_file],
      # Add transitive files from dependencies manually.
      transitive_files = transitive_runfiles,
      # Collect runfiles from the common locations: transitively from srcs,
      # deps and data attributes.
      collect_default = True,
  )
  # Add a field named "runfiles" to the DefaultInfo provider in order to actually
  # create the symlink tree.
  return [DefaultInfo(runfiles=runfiles)]
```

Note that non-executable rule outputs can also have runfiles. For example, a
library might need some external files during runtime, and every dependent
binary should know about them.

Also note that if an action uses an executable, the executable's runfiles can
be used when the action executes.

Normally, the relative path of a file in the runfiles tree is the same as the
relative path of that file in the source tree or generated output tree. If these
need to be different for some reason, you can specify the `root_symlinks` or
`symlinks` arguments.  The `root_symlinks` is a dictionary mapping paths to
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
targets.

## Requesting output files

A single target can have several output files. When a `bazel build` command is
run, some of the outputs of the targets given to the command are considered to
be *requested*. Bazel only builds these requested files and the files that they
directly or indirectly depend on. (In terms of the action graph, Bazel only
executes the actions that are reachable as transitive dependencies of the
requested files.)

Every target has a set of *default outputs*, which are the output files that
normally get requested when that target appears on the command line. For
example, a target `//pkg:foo` of `java_library` type has in its default outputs
a file `foo.jar`, which will be built by the command `bazel build //pkg:foo`.

Any predeclared output can be explicitly requested on the command line. This can
be used to build outputs that are not default outputs, or to build some but not
all default outputs. For example, `bazel build //pkg:foo_deploy.jar` and
`bazel build //pkg:foo.jar` will each just build that one file (along with
its dependencies). See an [example](https://github.com/bazelbuild/examples/blob/master/rules/implicit_output/hash.bzl)
of a rule with non-default predeclared outputs.

In addition to default outputs, there are *output groups*, which are collections
of output files that may be requested together. For example, if a target
`//pkg:mytarget` is of a rule type that has a `debug_files` output group, these
files can be built by running
`bazel build //pkg:mytarget --output_groups=debug_files`. See the [command line
reference](https://docs.bazel.build/versions/master/command-line-reference.html#flag--output_groups)
for details on the `--output_groups` argument. Since non-predeclared outputs
don't have labels, they can only be requested by appearing in the default
outputs or an output group.

You can specify the default outputs and output groups of a rule by returning the
[`DefaultInfo`](lib/DefaultInfo.html) and
[`OutputGroupInfo`](lib/OutputGroupInfo.html) providers from its implementation
function.

```python
def _myrule_impl(ctx):
  name = ...
  binary = ctx.actions.declare_file(name)
  debug_file = ctx.actions.declare_file(name + ".pdb")
  # ... add actions to generate these files
  return [DefaultInfo(files = depset([binary])),
          OutputGroupInfo(debug_files = depset([debug_file]),
                          all_files = depset([binary, debug_file]))]
```

These providers can also be retrieved from dependencies using the usual syntax
`<target>[DefaultInfo]` and `<target>[OutputGroupInfo]`, where `<target>` is a
`Target` object.

Note that even if a file is in the default outputs or an output group, you may
still want to return it in a custom provider in order to make it available in a
more structured way. For instance, you could pass headers and sources along in
separate fields of your provider.

## Code coverage instrumentation

A rule can use the `instrumented_files` provider to provide information about
which files should be measured when code coverage data collection is enabled:

```python
def _rule_implementation(ctx):
  ...
  return struct(instrumented_files = struct(
      # Optional: File extensions used to filter files from source_attributes.
      # If not provided, then all files from source_attributes will be
      # added to instrumented files, if an empty list is provided, then
      # no files from source attributes will be added.
      extensions = ["ext1", "ext2"],
      # Optional: Attributes that contain source files for this rule.
      source_attributes = ["srcs"],
      # Optional: Attributes for dependencies that could include instrumented
      # files.
      dependency_attributes = ["data", "deps"]))
```

[ctx.configuration.coverage_enabled](lib/configuration.html#coverage_enabled) notes
whether coverage data collection is enabled for the current run in general
(but says nothing about which files specifically should be instrumented).
If a rule implementation needs to add coverage instrumentation at
compile-time, it can determine if its sources should be instrumented with
[ctx.coverage_instrumented](lib/ctx.html#coverage_instrumented):

```python
# Are this rule's sources instrumented?
if ctx.coverage_instrumented():
  # Do something to turn on coverage for this compile action
```

Note that function will always return false if `ctx.configuration.coverage_enabled` is
false, so you don't need to check both.

If the rule directly includes sources from its dependencies before compilation
(e.g. header files), it may also need to turn on compile-time instrumentation
if the dependencies' sources should be instrumented. In this case, it may
also be worth checking `ctx.configuration.coverage_enabled` so you can avoid looping
over dependencies unnecessarily:

```python
# Are this rule's sources or any of the sources for its direct dependencies
# in deps instrumented?
if ctx.configuration.coverage_enabled:
    if (ctx.coverage_instrumented() or
        any([ctx.coverage_instrumented(dep) for dep in ctx.attr.deps]):
        # Do something to turn on coverage for this compile action
```

## Executable rules and test rules

Executable rules define targets that can be invoked by a `bazel run` command.
Test rules are a special kind of executable rule whose targets can also be
invoked by a `bazel test` command. Executable and test rules are created by
setting the respective [`executable`](lib/globals.html#rule.executable) or
[`test`](lib/globals.html#rule.test) argument to true when defining the rule.

Test rules (but not necessarily their targets) must have names that end in
`_test`. Non-test rules must not have this suffix.

Both kinds of rules must produce an executable output file (which may or may not
be predeclared) that will be invoked by the `run` or `test` commands. To tell
Bazel which of a rule's outputs to use as this executable, pass it as the
`executable` argument of a returned [`DefaultInfo`](lib/DefaultInfo.html)
provider.

The action that generates this file must set the executable bit on the file. For
a `ctx.actions.run()` or `ctx.actions.run_shell()` action this should be done by
the underlying tool that is invoked by the action. For a `ctx.actions.write()`
action it is done by passing the argument `is_executable=True`.

As legacy behavior, executable rules have a special `ctx.outputs.executable`
predeclared output. This file serves as the default executable if you do not
specify one using `DefaultInfo`; it must not be used otherwise. This output
mechanism is deprecated because it does not support customizing the executable
file's name at analysis time.

See examples of an [executable rule](https://github.com/bazelbuild/examples/blob/master/rules/executable/fortune.bzl)
and a [test rule](https://github.com/bazelbuild/examples/blob/master/rules/test_rule/line_length.bzl).

Test rules inherit the following attributes: `args`, `flaky`, `local`,
`shard_count`, `size`, `timeout`. The defaults of inherited attributes cannot be
changed, but you can use a macro with default arguments:

```python
def example_test(size="small", **kwargs):
  _example_test(size=size, **kwargs)

_example_test = rule(
 ...
)
```

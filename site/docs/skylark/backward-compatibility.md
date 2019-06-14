---
layout: documentation
title: Legacy Incompatible Changes List
---


# Legacy Incompatible Changes List

Legacy, partial list of [backward-incompatible changes](../backward-compatibility.md).

Full, authorative list of incompatible changes is [GitHub issues with
"incompatible-change" label](https://github.com/bazelbuild/bazel/issues?q=is%3Aissue+label%3Aincompatible-change)


General Starlark

*   [Dictionary concatenation](#dictionary-concatenation)
*   [Load must appear at top of file](#load-must-appear-at-top-of-file)
*   [Depset is no longer iterable](#depset-is-no-longer-iterable)
*   [Depset union](#depset-union)
*   [String is no longer iterable](#string-is-no-longer-iterable)
*   [Integer division operator is //](#integer-division-operator-is-)
*   [Package name is a function](#package-name-is-a-function)
*   [FileType is deprecated](#filetype-is-deprecated)
*   [Static Name Resolution](#static-name-resolution)
*   [Load label cannot cross package boundaries](#load-label-cannot-cross-package-boundaries)

Starlark Rules

*   [New actions API](#new-actions-api)
*   [New args API](#new-args-api)
*   [Disable output group field on Target](#disable-output-group-field-on-target)
*   [Disable default parameter of output attributes](#disable-default-parameter-of-output-attributes)
*   [Disallow tools in action inputs](#disallow-tools-in-action-inputs)
*   [Expand directories in Args](#expand-directories-in-args)
*   [Disable late bound option defaults](#disable-late-bound-option-defaults)
*   [Disallow `cfg = "data"`](#disallow-cfg--data)

Objc

*   [Disable objc provider resources](#disable-objc-provider-resources)

External repositories

*   [Remove native git repository](#remove-native-git-repository)
*   [Remove native http archive](#remove-native-http-archive)
*   [Remove native maven jar](#remove-native-maven-jar)

Java

*   [New-style JavaInfo constructor](#new-style-java_info)

Misc

*   [Disable InMemory Tools Defaults Package](#disable-inmemory-tools-defaults-package)

C++

*   [Disable depsets in C++ toolchain API in user
    flags](#disable-depsets-in-c-toolchain-api-in-user-flags)
*   [Disallow using CROSSTOOL to select the cc_toolchain label](#disallow-using-crosstool-to-select-the-cc_toolchain-label)
*   [Disallow using C++ Specific Make Variables from the configuration](#disallow-using-c-specific-make-variables-from-the-configuration)
*   [Disable legacy C++ configuration API](#disable-legacy-c-configuration-api)
*   [Disable legacy C++ toolchain API](#disable-legacy-c-toolchain-api)


### Dictionary concatenation

We are removing the `+` operator on dictionaries. This includes the `+=` form
where the left-hand side is a dictionary. This is done to improve compatibility
with Python. A possible workaround is to use the `.update` method instead.

*   Flag: `--incompatible_disallow_dict_plus`
*   Default: `true`
*   Tracking issue: [#6461](https://github.com/bazelbuild/bazel/issues/6461)

### Load must appear at top of file

Previously, the `load` statement could appear anywhere in a `.bzl` file so long
as it was at the top level. With this change, for `.bzl` files, `load` must
appear at the beginning of the file, i.e. before any other non-`load` statement.

*   Flag: `--incompatible_bzl_disallow_load_after_statement`
*   Default: `true`
*   Tracking issue: [#5815](https://github.com/bazelbuild/bazel/issues/5815)


### Depset is no longer iterable

When the flag is set to true, `depset` objects are not treated as iterable. This
prohibits directly iterating over depsets in `for` loops, taking its size via
`len()`, and passing it to many functions such as `list`, `tuple`, `min`, `max`,
`sorted`, `all`, and `any`. It does not prohibit checking for emptiness by
converting the depset to a boolean.

The goal of this change is to avoid accidental iteration on `depset`, which can
be [expensive](performance.md#avoid-calling-depsetto-list). If you really need
to iterate over a depset, you can call the `.to_list()` method to obtain a
flattened list of its contents.

``` python
deps = depset()
[x.path for x in deps]  # deprecated
[x.path for x in deps.to_list()]  # recommended

sorted(deps)  # deprecated
sorted(deps.to_list())  # recommended
```

*   Flag: `--incompatible_depset_is_not_iterable`
*   Default: `false`
*   Tracking issue: [#5816](https://github.com/bazelbuild/bazel/issues/5816)


### Depset union

To merge two sets, the following examples used to be supported, but are now
deprecated:

``` python
depset1 + depset2  # deprecated
depset1 | depset2  # deprecated
depset1.union(depset2)  # deprecated
```

The recommended solution is to use the `depset` constructor:

``` python
depset(transitive = [depset1, depset2])
```

See the [`depset documentation`](depsets.md) for more information.

*   Flag: `--incompatible_depset_union`
*   Default: `false`
*   Tracking issue: [#5817](https://github.com/bazelbuild/bazel/issues/5817)


### String is no longer iterable

When the flag is set to true, `string` objects are not treated as iterable. This
affects `for` loops and many functions, e.g. `list`, `tuple`, `min`, `max`,
`sorted`, `all`, and `any`. String iteration has been a source of errors and
confusion, such as this error:

``` python
def my_macro(name, srcs):
  for src in srcs:
    # do something with src

# equivalent to: my_macro("hello", ["f", "o", "o", ".", "c", "c"])
my_macro(
  name = "hello",
  srcs = "foo.cc",
)
```

String indexing and `len` are still allowed. If you need to iterate over a
string, you may explicitly use:

``` python
my_string = "hello world"
for i in range(len(my_string)):
  char = my_string[i]
  # do something with char
```

*   Flag: `--incompatible_string_is_not_iterable`
*   Default: `true`
*   Tracking issue: [#5830](https://github.com/bazelbuild/bazel/issues/5830)


### Package name is a function

The current package name should be retrieved by calling `package_name()` in
BUILD files or `native.package_name()` in .bzl files. The old way of referring
to the magic `PACKAGE_NAME` variable bends the language since it is neither a
parameter, local variable, nor global variable.

Likewise, the magic `REPOSITORY_NAME` variable is replaced by
`repository_name()` and `native.repository_name()`. Both deprecations use the
same flag.

*   Flag: `--incompatible_package_name_is_a_function`
*   Default: `true`
*   Tracking issue: [#5827](https://github.com/bazelbuild/bazel/issues/5827)


### FileType is deprecated

The [FileType](lib/FileType.html) function is going away. The main use-case was
as an argument to the [rule function](lib/globals.html#rule). It's no longer
needed, you can simply pass a list of strings to restrict the file types the
rule accepts.

*   Flag: `--incompatible_disallow_filetype`
*   Default: `true`
*   Tracking issue: [#5831](https://github.com/bazelbuild/bazel/issues/5831)


### New actions API

This change removes the old methods for registering actions within rules, and
requires that you use the new methods instead. The deprecated methods and their
replacements are as follows.

*   `ctx.new_file(...)` --> `ctx.actions.declare_file(...)`
*   `ctx.experimental_new_directory(...)` -->
    `ctx.actions.declare_directory(...)`
*   `ctx.action(...)` --> either `ctx.actions.run(...)` or
    `ctx.actions.run_shell(...)`
*   `ctx.file_action(...)` --> `ctx.actions.write(...)`
*   `ctx.empty_action(...)` --> `ctx.actions.do_nothing(...)`
*   `ctx.template_action(...)` --> `ctx.actions.expand_template(...)`

<!-- filler comment, needed by Markdown to separate the lists -->

*   Flag: `--incompatible_new_actions_api`
*   Default: `false`
*   Tracking issue: [#5825](https://github.com/bazelbuild/bazel/issues/5825)


### New args API

The [Args](lib/Args.html) object returned by `ctx.actions.args()` has dedicated
methods for appending the contents of a list or depset to the command line.
Previously these use cases were lumped into its [`add()`](lib/Args.html#add)
method, resulting in a more cluttered API.

With this flag, `add()` only works for scalar values, and its deprecated
parameters are disabled. To add many arguments at once you must use `add_all()`
or `add_joined()` instead.

*   Flag: `--incompatible_disallow_old_style_args_add`
*   Default: `true`
*   Tracking issue: [#5822](https://github.com/bazelbuild/bazel/issues/5822)


### Disable objc provider resources

This flag disables certain deprecated resource fields on
[ObjcProvider](lib/ObjcProvider.html).

*   Flag: `--incompatible_objc_provider_resources`
*   Default: `false`


### Disable output group field on Target

This flag disables the `output_group` field on the `Target` Starlark type.
Use `OutputGroupInfo` instead.

For example, replace:

```python
dep_bin = ctx.attr.dep.output_group.bin
```

with:

```python
dep_bin = ctx.attr.dep[OutputGroupInfo].bin
```

*   Flag: `--incompatible_no_target_output_group`
*   Default: `false`
*   Tracking issue: [#6241](https://github.com/bazelbuild/bazel/issues/6241)


### Disable default parameter of output attributes

This flag disables the `default` parameter on `attr.output` and
`attr.output_list`. Use Starlark macros to specify defaults for these attributes
instead.

For example, replace:

```python
my_rule = rule(
    ...
    attrs = {"out" : attr.output(default = "foo.txt")}
    ...
```

with:

```python
# myrule.bzl
my_rule = rule(
    ...
    attrs = {"out" : attr.output()}
    ...

# mymacro.bzl
load(":myrule.bzl", _my_rule = "my_rule")

def my_rule(name):
    _my_rule(
        name = name,
        output = "%s_out.txt" % name
    )
```

The previous `default` parameter of these attribute types was severely
bug-prone, as two targets of the same rule would be unable to exist in the same
package under default behavior. (Two targets both generating `foo.txt` in the
same package would conflict.)

*   Flag: `--incompatible_no_output_attr_default`
*   Default: `false`
*   Tracking issue: [#6241](https://github.com/bazelbuild/bazel/issues/6241)


### Remove native git repository

When set, the native `git_repository` and `new_git_repository` rules are
disabled. The Starlark versions

```python
load("@bazel_tools//tools/build_defs/repo:git.bzl",
     "git_repository", "new_git_repository")
```

should be used instead. These are drop-in replacements of the corresponding
native rules, however with the additional requirement that all label arguments
be provided as a fully qualified label (usually starting with `@//`),
for example: `build_file = "@//third_party:repo.BUILD"`.

*   Flag: `--incompatible_remove_native_git_repository`
*   Default: `true`


### Remove native http archive

When set, the native `http_archive` and all related rules are disabled.
The Starlark version

```python
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
```

should be used instead. This is a drop-in replacement, however with the
additional requirement that all label arguments be provided as
fully qualified labels (usually starting with `@//`). The Starlark `http_archive`
is also a drop-in replacement for the native `new_http_archive` (with
the same proviso). `http.bzl` also
provides `http_jar` and `http_file` (the latter only supports the `urls`
parameter, not `url`).

*   Flag: `--incompatible_remove_native_http_archive`
*   Default: `true`

### Remove native maven jar

When set, the native `maven_jar` rule is disabled. The Starlark version

```python
load("@bazel_tools//tools/build_defs/repo:java.bzl", "java_import_external")
```

or the convenience wrapper

```python
load("@bazel_tools//tools/build_defs/repo:jvm.bzl", "jvm_maven_import_external")
```

should be used instead. These rules are more reliable and offer additional
functionality over the native `maven_jar` rule. In addition to downloading
the jars, they allow defining the jar's dependencies. They also enable
downloading src-jars.

Given a `WORKSPACE` file that looks like the following:

```python
maven_jar(
    name = "truth",
    artifact = "com.google.truth:truth:0.30",
    sha1 = "9d591b5a66eda81f0b88cf1c748ab8853d99b18b",
)
```

It will need to look like this after updating:
```python
load("@bazel_tools//tools/build_defs/repo:jvm.bzl", "jvm_maven_import_external")
jvm_maven_import_external(
    name = "truth",
    artifact = "com.google.truth:truth:0.30",
    artifact_sha256 = "59721f0805e223d84b90677887d9ff567dc534d7c502ca903c0c2b17f05c116a",
    server_urls = ["http://central.maven.org/maven2"],
    licenses = ["notice"],  # Apache 2.0
)
```

Notably
*   the `licenses` attribute is mandatory
*   sha1 is no longer supported, only sha256 is
*   the `server_urls` attribute is mandatory. If your `maven_jar` rule
    did not specify a url then you should use the default server
    ("http://central.maven.org/maven2"). If your rule did specify a url then
    keep using that one.

Documentation for the rule is
[here](https://source.bazel.build/bazel/+/master:tools/build_defs/repo/java.bzl;l=15).

*   Flag: `--incompatible_remove_native_maven_jar`
*   Default: `false`

### New-style JavaInfo constructor

When set, `java_common.create_provider` and certain arguments to `JavaInfo` are deprecated. The
deprecated arguments are: `actions`, `sources`, `source_jars`, `use_ijar`, `java_toolchain`,
and `host_javabase`.

Example migration from `create_provider`:

```python
# Before
provider = java_common.create_provider(
    ctx.actions,
    compile_time_jars = [output_jar],
    use_ijar = True,
    java_toolchain = ctx.attr._java_toolchain,
    transitive_compile_time_jars = transitive_compile_time,
    transitive_runtime_jars = transitive_runtime_jars,
)

# After
compile_jar = java_common.run_ijar(
    ctx.actions,
    jar = output_jar,
    target_label = ctx.label,
    java_toolchain = ctx.attr._java_toolchain,
)
provider = JavaInfo(
    output_jar = output_jar,
    compile_jar = compile_jar,
    deps = deps,
    runtime_deps = runtime_deps,
)
```

Example migration from deprecated `JavaInfo` arguments:

```python
# Before
provider = JavaInfo(
    output_jar = my_jar,
    use_ijar = True,
    sources = my_sources,
    deps = my_compile_deps,
    runtime_deps = my_runtime_deps,
    actions = ctx.actions,
    java_toolchain = my_java_toolchain,
    host_javabase = my_host_javabase,
)

# After
my_ijar = java_common.run_ijar(
    ctx.actions,
    jar = my_jar,
    target_label = ctx.label,
    java_toolchain, my_java_toolchain,
)
my_source_jar = java_common.pack_sources(
    ctx.actions,
    sources = my_sources,
    java_toolchain = my_java_toolchain,
    host_javabase = my_host_javabase,
)
provider = JavaInfo(
    output_jar = my_jar,
    compile_jar = my_ijar,
    source_jar = my_source_jar,
    deps = my_compile_deps,
    runtime_deps = my_runtime_deps,
)
```

### Disallow tools in action inputs

A tool is an input coming from an attribute of type `label`
where the attribute has been marked `executable = True`. In order for an action
to run a tool, it needs access to its runfiles.

Under the old API, tools are passed to `ctx.actions.run()` and
`ctx.actions.run_shell()` via their `inputs` parameter. Bazel scans this
argument (which may be a large depset) to find all the inputs that are tools,
and adds their runfiles automatically.

In the new API, tools are instead passed to a dedicated `tools` parameter. The
`inputs` are not scanned. If a tool is accidentally put in `inputs` instead of
`tools`, the action will fail during the execution phase with an error due to
missing runfiles. This may be somewhat cryptic.

To support a gradual transition, all actions with a `tools` argument are opted
into the new API, while all actions without a `tools` argument still follow the
old one. In the future (when this flag is removed), all actions will use the new
API unconditionally.

This flag turns on a safety check that is useful for migrating existing code.
The safety check applies to all actions that do not have a `tools` argument. It
scans the `inputs` looking for tools, and if it finds any, it raises an error
during the analysis phase that clearly identifies the offending tools.

In the rare case that your action requires a tool as input, but does not
actually run the tool and therefore does not need its runfiles, the safety check
will fail even though the action would have succeeded. In this case, you can
bypass the check by adding a (possibly empty) `tools` argument to your action.
Note that once an action has been modified to take a `tools` argument, you will
no longer get helpful analysis-time errors for any remaining tools that should
have been migrated from `inputs`.


*   Flag: `--incompatible_no_support_tools_in_action_inputs`
*   Default: `false`


### Expand directories in Args

Previously, directories created by
[`ctx.actions.declare_directory`](lib/actions.html#declare_directory) expanded
to the path of the directory when added to an [`Args`](lib/Args.html) object.

With this flag enabled, directories are instead replaced by the full file
contents of that directory when passed to `args.add_all()` or
`args.add_joined()`. (Directories may not be passed to `args.add()`.)

If you want the old behavior on a case-by-case basis (perhaps your tool can
handle directories on the command line), you can pass `expand_directories=False`
to the `args.add_all()` or `args.add_joined()` call.

```
d = ctx.action.declare_directory("dir")
# ... Some action runs and produces ["dir/file1", "dir/file2"] ...
f = ctx.action.declare_file("file")
args = ctx.action.args()
args.add_all([d, f])
#  -> Used to expand to ["dir", "file"]
#     Now expands to ["dir/file1", "dir/file2", "file"]
```

*   Flag: `--incompatible_expand_directories`
*   Default: `false`


### Static Name Resolution

When the flag is set, use a saner way to resolve variables. The previous
behavior was buggy in a number of subtle ways. See [the
proposal](https://github.com/bazelbuild/proposals/blob/master/docs/2018-06-18-name-resolution.md)
for background and examples.

The proposal is not fully implemented yet.

*   Flag: `--incompatible_static_name_resolution`
*   Default: `true`
*   Tracking issue: [#5637](https://github.com/bazelbuild/bazel/issues/5637)


### Disallow transitive loads

When the flag is set, `load` can only import symbols that were explicitly
defined in the target file, using either `=` or `def`.

When the flag is unset (legacy behavior), `load` may also import symbols that
come from other `load` statements.

In other words, the `x` below is exported only if the flag is unset:

```python
load(":file.bzl", "x")

y = 1
```

*   Flag: `--incompatible_no_transitive_loads`
*   Default: `true`
*   Introduced in: `0.19.0`
*   Tracking issue: [#5636](https://github.com/bazelbuild/bazel/issues/5636)


### Disable InMemory Tools Defaults Package

If false, Bazel constructs an in-memory `//tools/defaults` package based on the
command line options. If true, `//tools/defaults:*` is resolved from file system
as a regular package.

*   Flag: `--incompatible_disable_tools_defaults_package`
*   Default: `false`

#### Motivation:

`//tools/default` was initially created as virtual in-memory package. It
generates content dynamically based on current configuration. There is no need
of having `//tools/defaults` any more as LateBoundAlias can do dynamic
configuration-based label resolving.  Also, having `//tools/default` makes
negative impact on performance, and introduces unnecessary code complexity.

All references to `//tools/defaults:*` targets should be removed or replaced
to corresponding target in `@bazel_tools//tools/jdk:` and
`@bazel_tools//tools/cpp:` packages.

#### Scope of changes and impact:

Targets in `//tools/default` will not exist any more. If you have any references
inside your BUILD or *.bzl files to any of its, then bazel will fail to resolve.

#### Migration plan:

Please replace all occurrences:

*   `//tools/defaults:jdk`
    *   by `@bazel_tools//tools/jdk:current_java_runtime`
    *   or/and `@bazel_tools//tools/jdk:current_host_java_runtime`
*   `//tools/defaults:java_toolchain`
    *   by `@bazel_tools//tools/jdk:current_java_toolchain`
*   `//tools/defaults:crosstool`
    *   by `@bazel_tools//tools/cpp:current_cc_toolchain`
    *   or/and `@bazel_tools//tools/cpp:current_cc_host_toolchain`
    *   if you need reference to `libc_top`, then `@bazel_tools//tools/cpp:current_libc_top`

These targets will not be supported any more:

*   `//tools/defaults:coverage_report_generator`
*   `//tools/defaults:coverage_support`

### Disable late bound option defaults

If true, Bazel will stop retrieving the value of `compiler` from the cpp configuration when
`--compiler` is not specified. This will cause a `config_setting` that have
`values = {"compiler": "x"}` to not work properly when `--compiler` is not specified at command
line.

The former behavior can be achieved by changing the `config_setting` to use
`flag_values = {"@bazel_tools//tools/cpp:compiler": "x"}` instead:

```python
# Before
config_setting(
    name = "cpu_x_compiler_y",
    values = {
        "cpu": "x",
        "compiler": "y",
    },
)

# After
config_setting(
    name = "cpu_x_compiler_y",
    values = {
        "cpu": "x",
    },
    flag_values = {
        "@bazel_tools//tools/cpp:compiler": "y",
    },
)
```

*   Flag: `--incompatible_disable_late_bound_option_defaults`
*   Default: `false`
*   Introduced in: `0.18.0`
*   Tracking issue: [#6384](https://github.com/bazelbuild/bazel/issues/6384)

### Disable depsets in C++ toolchain API in user flags

If true, Bazel will no longer accept depsets in `user_compile_flags` for
[create\_compile\_variables](../skylark/lib/cc_common.html#create_compile_variables),
and in `user_link_flags` for
[create\_link\_variables](../skylark/lib/cc_common.html#create_link_variables).
Use plain lists instead.

*   Flag: `--incompatible_disable_depset_in_cc_user_flags`
*   Default: `false`
*   Introduced in: `0.18.0`
*   Tracking issue: [#6383](https://github.com/bazelbuild/bazel/issues/6383)

### Disallow using CROSSTOOL to select the cc_toolchain label

Currently Bazel selects the `cc_toolchain` to use from the `toolchains`
dictionary attribute of `cc_toolchain_suite`. The key it uses is constructed
the following way:

*   If `--compiler` option is specified, the key is `--cpu|--compiler`. Bazel
     errors out if the entry doesn't exist.
*   If `--compiler` option was not specified on command line, Bazel checks if
     an entry with the key `--cpu` exists, and uses it if it does. If such an
     entry doesn't exist, it loops through the `default_toolchain` list in the
     CROSSTOOL file, selects the first one that matches the `--cpu` option,
     finds the `CToolchain` whose identifier matches the
     `default_toolchain.toolchain_identifier` field, and then uses the key
     `CToolchain.targetCpu|Ctoolchain.compiler`. It errors out if the entry
     doesn't exist.

We're making selection of the `cc_toolchain` label independent of the
CROSSTOOL file: when the flag is set to True, Bazel will no longer loop
through the `default_toolchain` list in order to construct a key for selecting
a `cc_toolchain` label from `cc_toolchain_suite.toolchains`, but throw an error
instead.

In order to not be affected by this change, one should add entries in the
`cc_toolchain_suite.toolchains` for the potential values of `--cpu`:

```python
# Before
cc_toolchain_suite(
    toolchains = {
        "cpu1|compiler1": ":cc_toolchain_label1",
        "cpu2|compiler2": ":cc_tolchain_label2",
    }
)

# After
cc_toolchain_suite(
    toolchains = {
        "cpu1|compiler1": ":cc_toolchain_label1",
        "cpu2|compiler2": ":cc_toolchain_label2",
        "cpu1": ":cc_toolchain_label3",
        "cpu2": ":cc_tolchain_label4",
    }
)
```

Before, it could happen that the same `cc_toolchain` is used with multiple
`CToolchain`s from the CROSSTOOL through `default_toolchain`s. This is no longer
allowed, each `cc_toolchain` must point to at most one `CToolchain` by:

* (preferable) specifying `cc_toolchain.toolchain_identifier` equal to
  `CToolchain.toolchain_identifier`
* (deprecated, but still supported, doesn't work without specifying `compiler`)
  specifying `cc_toolchain.cpu` and `cc_toolchain.compiler` fields that match
  `CToolchain.target_cpu` and `CToolchain.compiler` respectively.
* (deprecated, but still supported, doesn't work with
  [platforms](https://www.bazel.build/roadmaps/platforms.html)) Relying on
  `--cpu` and `--compiler` options.

Using `cc_toolchain.toolchain_identifier` will save you one migration in the
future.

*   Flag: `--incompatible_disable_cc_toolchain_label_from_crosstool_proto`
*   Default: `false`
*   Introduced in: `0.18.0`
*   Tracking issue: [#6382](https://github.com/bazelbuild/bazel/issues/6382)

### Disallow using C++ Specific Make Variables from the configuration

Currently Bazel allows rule authors to access certain Make variables that are
implicitly provided to every rule by the CppConfiguration. This causes every
target to implicitly depend on CppConfiguration, which creates an undesirable
number of extra, unused, dependencies.

We are removing the implicit provision of these Make variables, and requiring
rules and targets that use these Make variables to explicitly depend on a
C++ toolchain in order to access them.

The list of Make variables is:

* CC
* AR
* NM
* LD
* OBJCOPY
* STRIP
* GCOVTOOL
* GLIBC\_VERSION
* C\_COMPILER
* CROSSTOOLTOP
* ABI\_GLIBC\_VERSION
* ABI

In order to not be affected by this change, add a C++ toolchain to the
`toolchains` attribute for targets, or to the`_toolchains` attribute for
Starlark rules. The best choice for this value is
the alias target `@bazel_tools//tools/cpp:current_cc_toolchain`, which will
always resolve to the currently selected C++ toolchain.

Genrules will still have access to these Make variables for the time
being because that information is plumbed not through CppConfiguration, but
through an implicit dependency on the C++ toolchain. That will also be
removed at some point in the future, so it's considered good practice to add an
explicit dependency on the toolchain as demonstrated below.

For genrules and other targets using C++ Make Variables:

```python
# Before
genrule(
    cmd = "$(STRIP) file-to-be-stripped.o",
)

# After
genrule(
    cmd = "$(STRIP) file-to-be-stripped.o",
    toolchains = ["@bazel_tools//tools/cpp:current_cc_toolchain"],
)
```

For Starlark rules using C++ Make Variables:

```python
# Before
def _impl(ctx):
    strip = ctx.var["STRIP"]
    ...

my_rule = rule(
    implementation = _impl,
    attrs = {
    },
)

# After
def _impl(ctx):
    strip = ctx.var["STRIP"]
    ...

my_rule = rule(
    implementation = _impl,
    attrs = {
        "_toolchains": attr.label_list(default = [Label("@bazel_tools//tools/cpp:current_cc_toolchain")]),
    },
)
```
*   Flag: `--incompatible_disable_cc_configuration_make_variables`
*   Default: `false`
*   Introduced in: `0.18.0`
*   Tracking issue: [#6381](https://github.com/bazelbuild/bazel/issues/6381)

### Disable legacy C++ configuration API

**You might want to migrate for this flag together with
`--incompatible_disable_legacy_flags_cc_toolchain_api` in a single go.
Migration instructions for
`--incompatible_disable_legacy_cpp_toolchain_skylark_api` use an API that is
already deprecated by `--incompatible_disable_legacy_flags_cc_toolchain_api`**

This turns off legacy Starlark access to cc toolchain information via the
`ctx.fragments.cpp` fragment. Instead of declaring dependency on the `ctx.fragments.cpp` using the
`fragments` attribute declare a dependency on the `@bazel_tools//tools/cpp:current_cc_toolchain`
via implicit attribute named `_cc_toolchain` (see example below). Use `find_cpp_toolchain` from
`@bazel_tools//tools/cpp:toolchain_utils.bzl` to get the current C++ toolchain in the rule
  implementation.

```python
# Before
def _impl(ctx):
    ...
    ctx.fragments.cpp.compiler_options()

foo = rule(
    implementation = _impl,
    fragments = ["cpp"],
    ...
)

# After
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")

def _impl(ctx):
    ...
    cc_toolchain = find_cpp_toolchain(ctx)
    cc_toolchain.compiler_options()

foo = rule(
    implementation = _impl,
    attrs = {
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain")
        ),
    },
)
```

List of all legacy fields and their corresponding `cc_toolchain` alternative:

|`ctx.fragments.cpp` | `cc_toolchain`  |
|---|---|
| `ar_executable` |  `ar_executable()` |
| `built_in_include_directories` |  `built_in_include_directories` |
| `c_options` |  `c_options()` |
| `compiler` |  `compiler` |
| `compiler_executable` |  `compiler_executable()` |
| `compiler_options(unused_arg)` |  `compiler_options()` |
| `cpu` |  `cpu` |
| `cxx_options(unused_arg)` |  `cxx_options()` |
| `dynamic_link_options(unused_arg, bool)` |  `dynamic_link_options(bool)` |
| `fully_static_link_options(unused_arg, True)` |  `fully_static_link_options(True)` |
| `ld_executable` |  `ld_executable()` |
| `link_options` |  `link_options_do_not_use` |
| `mostly_static_link_options(unused_arg, bool)` |  `mostly_static_link_options(bool)` |
| `nm_executable` |  `nm_executable()` |
| `objcopy_executable` |  `objcopy_executable()` |
| `objdump_executable` |  `objdump_executable()` |
| `preprocessor_executable` |  `preprocessor_executable()` |
| `strip_executable` |  `strip_executable()` |
| `sysroot` |  `sysroot` |
| `target_gnu_system_name` |  `target_gnu_system_name` |
| `unfiltered_compiler_options(unused_arg)` |  `unfiltered_compiler_options(unused_arg)` |

If you use legacy Starlark API on `ctx.host_fragment.cpp`, let us know on
[the tracking bug for C++ migration to platforms](https://github.com/bazelbuild/bazel/issues/6516)
about your use case. The current plan is that host fragments will be removed.
To migrate, add an implicit rule attribute in the host configuration:

```python
"_host_cc_toolchain": attr.label(
    cfg = "host",
    default = Label("//tools/cpp:current_cc_toolchain"),
),
```

Then in your rules access the provider using:

```python
host_cc_toolchain = ctx.attr._host_cc_toolchain[cc_common.CcToolchainInfo]
```

*   Flag: `--incompatible_disable_legacy_cpp_toolchain_skylark_api`
*   Default: `false`
*   Introduced in: `0.18.0`
*   Tracking issue: [#6380](https://github.com/bazelbuild/bazel/issues/6380)

### Disable legacy C++ toolchain API

We have deprecated the `cc_toolchain` Starlark API returning legacy CROSSTOOL fields:

* c\_options
* compiler\_options
* cxx\_options
* dynamic\_link\_options
* fully\_static\_link\_options
* link\_options
* mostly\_static\_link\_options
* unfiltered\_compiler\_options

Use the new API from [cc_common](../skylark/lib/cc_common.html)

```python
# Before:
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")

def _impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    compiler_options = (
        cc_toolchain.compiler_options() +
        cc_toolchain.unfiltered_compiler_options([]) +
        ["-w", "-Wno-error"]
    )
    link_options = (
        ["-shared", "-static-libgcc"] +
        cc_toolchain.mostly_static_link_options(True) +
        ["-Wl,-whole-archive"] +
        [l.path for l in libs] +
        ["-Wl,-no-whole-archive"] +
        cc_toolchain.link_options_do_not_use
    )

# After
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load(
    "@bazel_tools//tools/build_defs/cc:action_names.bzl",
    "CPP_LINK_DYNAMIC_LIBRARY_ACTION_NAME",
    "C_COMPILE_ACTION_NAME",
)

def _impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    compile_variables = cc_common.create_compile_variables(
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        user_compile_flags = depset(["-w", "-Wno-error"]),
    )
    compiler_options = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = C_COMPILE_ACTION_NAME,
        variables = compile_variables,
    )

    link_variables = cc_common.create_link_variables(
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        is_linking_dynamic_library = True,
        user_link_flags =
            ["-static-libgcc"] +
            ["-Wl,-whole-archive"] +
            [lib.path for lib in libs] +
            ["-Wl,-no-whole-archive"],
    )
    link_flags = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = CPP_LINK_DYNAMIC_LIBRARY_ACTION_NAME,
        variables = link_variables,
    )
```

*   Flag: `--incompatible_disable_legacy_flags_cc_toolchain_api`
*   Default: `false`
*   Introduced in: `0.19.0`
*   Tracking issue: [#6434](https://github.com/bazelbuild/bazel/issues/6434)


### Disallow `cfg = "data"`

`cfg = "data"` is a no-op that incorrectly gives the impression dependencies under
it are built in a distinct "data" mode:

```python
my_rule = rule(
    ...
    "some_attr": attr.label_list(
        cfg = "data"  # This line does nothing
    )
)
```

The original semantics were unclear and were
[removed](https://github.com/bazelbuild/bazel/commit/8820d3ae601f229b72c61d2eb601b0e8e9b0111a#diff-ffd6930edbe7f2529b608c400fd19456)
in 0.16.0.

Because this syntax is non-functional and confusing, it's being removed outright
([#6153](https://github.com/bazelbuild/bazel/issues/6153)). The functionality it
implies will be provided by
[Starlark build configuration](https://github.com/bazelbuild/bazel/issues/5574).

When `--incompatible_disallow_data_transition=true`, builds using this syntax
fail with an error.

*   Flag: `--incompatible_disallow_data_transition`
*   Default: `true`
*   Introduced in: `0.16.0`
*   Tracking issue: [#6153](https://github.com/bazelbuild/bazel/issues/6153)


### Load label cannot cross package boundaries

Previously, the label argument to the `load` statement (the first argument) was
checked to ensure that it referenced an existing package but it was not checked
to ensure that it didn't cross a package boundary.

For example, in

```python
load("//a:b/c.bzl", "doesntmatter")
```

if this flag is set to `true`, the above statement will be in error if `//a/b`
is a package; in such a case, the correct way to reference `c.bzl` via a label
would be `//a/b:c.bzl`.

*   Flag: `--incompatible_disallow_load_labels_to_cross_package_boundaries`
*   Default: `false`
*   Tracking issue: [#6408](https://github.com/bazelbuild/bazel/issues/6408)

<!-- Add new options here -->

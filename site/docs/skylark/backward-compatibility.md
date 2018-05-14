---
layout: documentation
title: Backward Compatibility
---

# Backward Compatibility

Bazel is still in Beta and new releases may include backward incompatible
changes. As we make changes and polish the extension mechanism, old features
may be removed and new features that are not backward compatible may be added.

Backward incompatible changes are introduced gradually:

1.  The backward incompatible change is introduced behind a flag with its
    default value set to `false`.
2.  In a later release, the flag's default value will be set to `true`. You
    can still use the flag to disable the change.
3.  Then in a later release, the flag will be removed and you will no longer be
    able to disable the change.

To check if your code will be compatible with future releases you can:

*   Build your code with the flag `--all_incompatible_changes`. This flag
    enables all backward incompatible changes, and so you can ensure your code
    is compatible with upcoming changes.
*   Use boolean flags to enable/disable specific backward incompatible changes.

## Current backward incompatible changes

The following are the backward incompatible changes that are implemented and
guarded behind flags in the current release:

*   [Dictionary concatenation](#dictionary-concatenation)
*   [Load must appear at top of file](#load-must-appear-at-top-of-file)
*   [Depset is no longer iterable](#depset-is-no-longer-iterable)
*   [Depset union](#depset-union)
*   [String is no longer iterable](#string-is-no-longer-iterable)
*   [Integer division operator is //](#integer-division-operator-is)
*   [Package name is a function](#package-name-is-a-function)
*   [FileType is deprecated](#filetype-is-deprecated)
*   [New actions API](#new-actions-api)
*   [New args API](#new-args-api)
*   [Glob tracking](#glob-tracking)
*   [Disable objc provider resources](#disable-objc-provider-resources)
*   [Remove native git repository](#remove-native-git-repository)
*   [Remove native http archive](#remove-native-http-archive)
*   [New-style JavaInfo constructor](#new-style-java_info)


### Dictionary concatenation

We are removing the `+` operator on dictionaries. This includes the `+=` form
where the left-hand side is a dictionary. This is done to improve compatibility
with Python. A possible workaround is to use the `.update` method instead.

*   Flag: `--incompatible_disallow_dict_plus`
*   Default: `false`


### Load must appear at top of file

Previously, the `load` statement could appear anywhere in a `.bzl` file so long
as it was at the top level. With this change, for `.bzl` files, `load` must
appear at the beginning of the file, i.e. before any other non-`load` statement.

*   Flag: `--incompatible_bzl_disallow_load_after_statement`
*   Default: `false`


### Depset is no longer iterable

When the flag is set to true, `depset` objects are not treated as iterable. If
you need an iterable, call the `.to_list()` method. This affects `for` loops and
many functions, e.g. `list`, `tuple`, `min`, `max`, `sorted`, `all`, and `any`.
The goal of this change is to avoid accidental iteration on `depset`, which can
be expensive.

``` python
deps = depset()
[x.path for x in deps]  # deprecated
[x.path for x in deps.to_list()]  # recommended

sorted(deps)  # deprecated
sorted(deps.to_list())  # recommended
```

*   Flag: `--incompatible_depset_is_not_iterable`
*   Default: `false`


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
depset(transtive=[depset1, depset2])
```

See the [`depset documentation`](depsets.md) for more information.

*   Flag: `--incompatible_depset_union`
*   Default: `false`


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
*   Default: `false`


### Integer division operator is `//`

Integer division operator is now `//` instead of `/`. This aligns with
Python 3 and it highlights the fact it is a floor division.

```python
x = 7 / 2  # deprecated

x = 7 // 2  # x is 3
```

*   Flag: `--incompatible_disallow_slash_operator`
*   Default: `false`


### Package name is a function

The current package name should be retrieved by calling `package_name()` in
BUILD files or `native.package_name()` in .bzl files. The old way of referring
to the magic `PACKAGE_NAME` variable bends the language since it is neither a
parameter, local variable, nor global variable.

Likewise, the magic `REPOSITORY_NAME` variable is replaced by
`repository_name()` and `native.repository_name()`. Both deprecations use the
same flag.

*   Flag: `--incompatible_package_name_is_a_function`
*   Default: `false`


### FileType is deprecated

The [FileType](lib/FileType.html) function is going away. The main use-case was
as an argument to the [rule function](lib/globals.html#rule). It's no longer
needed, you can simply pass a list of strings to restrict the file types the
rule accepts.

*   Flag: `--incompatible_disallow_filetype`
*   Default: `false`


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


### New args API

The [Args](lib/Args.html) object returned by `ctx.actions.args()` has dedicated
methods for appending the contents of a list or depset to the command line.
Previously these use cases were lumped into its [`add()`](lib/Args.html#add)
method, resulting in a more cluttered API.

With this flag, `add()` only works for scalar values, and its deprecated
parameters are disabled. To add many arguments at once you must use `add_all()`
or `add_joined()` instead.

*   Flag: `--incompatible_disallow_old_style_args_add`
*   Default: `false`


### Glob tracking

When set, glob tracking is disabled. This is a legacy feature that we expect has
no user-visible impact.

*   Flag: `--incompatible_disable_glob_tracking`
*   Default: `true`


### Disable objc provider resources

This flag disables certain deprecated resource fields on
[ObjcProvider](lib/ObjcProvider.html).

*   Flag: `--incompatible_objc_provider_resources`
*   Default: `false`


### Remove native git repository

When set, the native `git_repository` rule is disabled. The Skylark version

```python
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
```

should be used instead.

*   Flag: `--incompatible_remove_native_git_repository`
*   Default: `false`


### Remove native http archive

When set, the native `http_archive` rule is disabled. The skylark version

```python
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
```

should be used instead.

*   Flag: `--incompatible_remove_native_http_archive`
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

<!-- Add new options here -->

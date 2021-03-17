---
layout: documentation
title: Configurable build attributes
category: getting-started
---

# Configurable Build Attributes

**_Configurable attributes_**, commonly known as [`select()`](
be/functions.html#select), is a Bazel feature that lets users toggle the values
of BUILD rule attributes at the command line.

This can be used, for example, for a multiplatform library that automatically
chooses the appropriate implementation for the architecture, or for a
feature-configurable binary that can be customized at build time.

## Example

```python
# myapp/BUILD

cc_binary(
    name = "mybinary",
    srcs = ["main.cc"],
    deps = select({
        ":arm_build": [":arm_lib"],
        ":x86_debug_build": [":x86_dev_lib"],
        "//conditions:default": [":generic_lib"],
    }),
)

config_setting(
    name = "arm_build",
    values = {"cpu": "arm"},
)

config_setting(
    name = "x86_debug_build",
    values = {
        "cpu": "x86",
        "compilation_mode": "dbg",
    },
)
```

This declares a `cc_binary` that "chooses" its deps based on the flags at the
command line. Specficially, `deps` becomes:

<table>
  <tr style="background: #E9E9E9; font-weight: bold">
    <td>Command</td>
    <td>deps =</td>
  </tr>
  <tr>
    <td><code>bazel build //myapp:mybinary --cpu=arm</code></td>
    <td><code>[":arm_lib"]</code></td>
  </tr>
  <tr>
    <td><code>bazel build //myapp:mybinary -c dbg --cpu=x86</code></td>
    <td><code>[":x86_dev_lib"]</code></td>
  </tr>
  <tr>
    <td><code>bazel build //myapp:mybinary --cpu=ppc</code></td>
    <td><code>[":generic_lib"]</code></td>
  </tr>
  <tr>
    <td><code>bazel build //myapp:mybinary -c dbg --cpu=ppc</code></td>
    <td><code>[":generic_lib"]</code></td>
  </tr>
</table>

`select()` serves as a placeholder for a value that will be chosen based on
*configuration conditions*, which are labels referencing [`config_setting`](be/general.html#config_setting)
targets. By using `select()` in a configurable attribute, the attribute
effectively adopts different values when different conditions hold.

Matches must be unambiguous: either exactly one condition must match or, if
multiple conditions match, one's `values` must be a strict superset of all
others'. For example, `values = {"cpu": "x86", "compilation_mode": "dbg"}` is an
unambiguous specialization of `values = {"cpu": "x86"}`. The built-in condition
[`//conditions:default`](#the-default-condition) automatically matches when
nothing else does.

While this example uses `deps`, `select()` works just as well on `srcs`,
`resources`, `cmd`, and most other attributes. Only a small number of attributes
are *non-configurable*, and these are clearly annotated. For example,
`config_setting`'s own
[`values`](be/general.html#config_setting.values) attribute is non-configurable.

## `select()` and dependencies

Certain attributes change the build parameters for all transitive dependencies
under a target. For example, `genrule`'s `tools` changes `--cpu` to the CPU of
the machine running Bazel (which, thanks to cross-compilation, may be different
than the CPU the target is built for). This is known as a
[configuration transition](https://docs.bazel.build/versions/master/glossary.html#transition).

Given

```python
#myapp/BUILD

config_setting(
    name = "arm_cpu",
    values = {"cpu": "arm"},
)

config_setting(
    name = "x86_cpu",
    values = {"cpu": "x86"},
)

genrule(
    name = "my_genrule",
    srcs = select({
        ":arm_cpu": ["g_arm.src"],
        ":x86_cpu": ["g_x86.src"],
    }),
    tools = select({
        ":arm_cpu": [":tool1"],
        ":x86_cpu": [":tool2"],
    }),
)

cc_binary(
    name = "tool1",
    srcs = select({
        ":arm_cpu": ["armtool.cc"],
        ":x86_cpu": ["x86tool.cc"],
    }),
)
```

running

```sh
$ bazel build //myapp:my_genrule --cpu=arm
```

on an `x86` developer machine binds the build to `g_arm.src`, `tool1`, and
`x86tool.cc`. Both of the `select`s attached to `my_genrule` use `my_genrule`'s
build parameters, which include `--cpu=arm`. The `tools` attribute changes
`--cpu` to `x86` for `tool1` and its transitive dependencies. The `select` on
`tool1` uses `tool1`'s build parameters, which include `--cpu=x86`.

## Configuration conditions

Each key in a configurable attribute is a label reference to a
[`config_setting`](be/general.html#config_setting) or
[`constraint_value`](be/platform.html#constraint_value).

`config_setting` is just a collection of
expected command line flag settings. By encapsulating these in a target, it's
easy to maintain "standard" conditions users can reference from multiple places.

`constraint_value` provides support for [multi-platform behavior](#platforms).


### Built-in flags

Flags like `--cpu` are built into Bazel: the build tool natively understands
them for all builds in all projects. These are specified with
[`config_setting`](be/general.html#config_setting)'s
[`values`](be/general.html#config_setting.values) attribute:

```python
config_setting(
    name = "meaningful_condition_name",
    values = {
        "flag1": "value1",
        "flag2": "value2",
        ...
    },
)
```

`flagN` is a flag name (without `--`, so `"cpu"` instead of `"--cpu"`). `valueN`
is the expected value for that flag. `:meaningful_condition_name` matches if
*every* entry in `values` matches. Order is irrelevant.

`valueN` is parsed as if it was set on the command line. This means:

*  `values = { "compilation_mode": "opt" }` matches `bazel build -c opt`
*  `values = { "force_pic": "true" }` matches `bazel build --force_pic=1`
*  `values = { "force_pic": "0" }` matches `bazel build --noforce_pic`

`config_setting` only supports flags that affect target behavior. For example,
[`--show_progress`](user-manual.html#flag--show_progress) isn't allowed because
it only affects how Bazel reports progress to the user. Targets can't use that
flag to construct their results. The exact set of supported flags isn't
documented. In practice, most flags that "make sense" work.

### Custom flags

You can model your own project-specific flags with
[Starlark build
settings][BuildSettings]. Unlike built-in flags, these are defined as build
targets, so Bazel references them with target labels.

These are triggered with [`config_setting`](be/general.html#config_setting)'s
[`flag_values`](be/general.html#config_setting.flag_values)
attribute:

```python
config_setting(
    name = "meaningful_condition_name",
    flag_values = {
        "//myflags:flag1": "value1",
        "//myflags:flag2": "value2",
        ...
    },
)
```

Behavior is the same as for [built-in flags](#built-in-flags). See [here](https://github.com/bazelbuild/examples/tree/master/rules/starlark_configurations/select_on_build_setting)
for a working example.

[`--define`](command-line-reference.html#flag--define)
is an alternative legacy syntax for custom flags (for example
`--define foo=bar`). This can be expressed either in the
[values](be/general.html#config_setting.values) attribute
(`values = {"define": "foo=bar"}`) or the
[define_values](be/general.html#config_setting.define_values) attribute
(`define_values = {"foo": "bar"}`). `--define` is only supported for backwards
compatibility. Prefer Starlark build settings whenever possible.

`values`, `flag_values`, and `define_values` evaluate independently. The
`config_setting` matches if all values across all of them match.

## The default condition

The built-in condition `//conditions:default` matches when no other condition
matches.

Because of the "exactly one match" rule, a configurable attribute with no match
and no default condition emits a `"no matching conditions"` error. This can
protect against silent failures from unexpected settings:

```python
# myapp/BUILD

config_setting(
    name = "x86_cpu",
    values = {"cpu": "x86"},
)

cc_library(
    name = "x86_only_lib",
    srcs = select({
        ":x86_cpu": ["lib.cc"],
    }),
)
```

```sh
$ bazel build //myapp:x86_only_lib --cpu=arm
ERROR: Configurable attribute "srcs" doesn't match this configuration (would
a default condition help?).
Conditions checked:
  //myapp:x86_cpu
```

For even clearer errors, you can set custom messages with `select()`'s
[`no_match_error`](#custom-error-messages) attribute.

## Platforms

While the ability to specify multiple flags on the command line provides
flexibility, it can also be burdensome to individually set each one every time
you want to build a target.
   [Platforms](platforms.html)
let you consolidate these into simple bundles.

```python
# myapp/BUILD

sh_binary(
    name = "my_rocks",
    srcs = select({
        ":basalt": ["pyroxene.sh"],
        ":marble": ["calcite.sh"],
        "//conditions:default": ["feldspar.sh"],
    }),
)

config_setting(
    name = "basalt",
    constraint_values = [
        ":black",
        ":igneous",
    ],
)

config_setting(
    name = "marble",
    constraint_values = [
        ":white",
        ":metamorphic",
    ],
)

# constraint_setting acts as an enum type, and constraint_value as an enum value.
constraint_setting(name = "color")
constraint_value(name = "black", constraint_setting = "color")
constraint_value(name = "white", constraint_setting = "color")
constraint_setting(name = "texture")
constraint_value(name = "smooth", constraint_setting = "texture")
constraint_setting(name = "type")
constraint_value(name = "igneous", constraint_setting = "type")
constraint_value(name = "metamorphic", constraint_setting = "type")

platform(
    name = "basalt_platform",
    constraint_values = [
        ":black",
        ":igneous",
    ],
)

platform(
    name = "marble_platform",
    constraint_values = [
        ":white",
        ":smooth",
        ":metamorphic",
    ],
)
```

The platform can be specified on the command line. It activates the
`config_setting`s that contain a subset of the platform's `constraint_values`,
allowing those `config_setting`s to match in `select()` expressions.

For example, in order to set the `srcs` attribute of `my_rocks` to `calcite.sh`,
you can simply run

```sh
bazel build //my_app:my_rocks --platforms=//myapp:marble_platform
```

Without platforms, this might look something like

```sh
bazel build //my_app:my_rocks --define color=white --define texture=smooth --define type=metamorphic
```

`select()` can also directly read `constraint_value`s:

```python
constraint_setting(name = "type")
constraint_value(name = "igneous", constraint_setting = "type")
constraint_value(name = "metamorphic", constraint_setting = "type")
sh_binary(
    name = "my_rocks",
    srcs = select({
        ":igneous": ["igneous.sh"],
        ":metamorphic" ["metamorphic.sh"],
    }),
)
```

This saves the need for boilerplate `config_setting`s when you only need to
check against single values.

Platforms are still under development. See the
[documentation](platforms-intro.html) for details.

## Combining `select()`s

`select` can appear multiple times in the same attribute:

```python
sh_binary(
    name = "my_target",
    srcs = ["always_include.sh"] +
           select({
               ":armeabi_mode": ["armeabi_src.sh"],
               ":x86_mode": ["x86_src.sh"],
           }) +
           select({
               ":opt_mode": ["opt_extras.sh"],
               ":dbg_mode": ["dbg_extras.sh"],
           }),
)
```

`select` cannot appear inside another `select`. If you need to nest `selects`
and your attribute takes other targets as values, use an intermediate target:

```python
sh_binary(
    name = "my_target",
    srcs = ["always_include.sh"],
    deps = select({
        ":armeabi_mode": [":armeabi_lib"],
        ...
    }),
)

sh_library(
    name = "armeabi_lib",
    srcs = select({
        ":opt_mode": ["armeabi_with_opt.sh"],
        ...
    }),
)
```

If you need a `select` to match when multiple conditions match, consider [AND
chaining](#and-chaining).

## OR chaining

Consider the following:

```python
sh_binary(
    name = "my_target",
    srcs = ["always_include.sh"],
    deps = select({
        ":config1": [":standard_lib"],
        ":config2": [":standard_lib"],
        ":config3": [":standard_lib"],
        ":config4": [":special_lib"],
    }),
)
```

Most conditions evaluate to the same dep. But this syntax is hard to read and
maintain. It would be nice to not have to repeat `[":standard_lib"]` multiple
times.

One option is to predefine the value as a BUILD variable:

```python
STANDARD_DEP = [":standard_lib"]

sh_binary(
    name = "my_target",
    srcs = ["always_include.sh"],
    deps = select({
        ":config1": STANDARD_DEP,
        ":config2": STANDARD_DEP,
        ":config3": STANDARD_DEP,
        ":config4": [":special_lib"],
    }),
)
```

This makes it easier to manage the dependency. But it still causes unnecessary
duplication.

For more direct support, use one of the following:

### <a name="selects-with-or"></a>`selects.with_or`

The
[with_or](https://github.com/bazelbuild/bazel-skylib/blob/master/docs/selects_doc.md#selectswith_or)
macro in [Skylib](https://github.com/bazelbuild/bazel-skylib)'s
[`selects`](https://github.com/bazelbuild/bazel-skylib/blob/master/docs/selects_doc.md)
module supports `OR`ing conditions directly inside a `select`:

```python
load("@bazel_skylib//lib:selects.bzl", "selects")
```

```python
sh_binary(
    name = "my_target",
    srcs = ["always_include.sh"],
    deps = selects.with_or({
        (":config1", ":config2", ":config3"): [":standard_lib"],
        ":config4": [":special_lib"],
    }),
)
```

### <a name="selects-config-setting-or-group"></a>`selects.config_setting_group`


The
[config_setting_group](https://github.com/bazelbuild/bazel-skylib/blob/master/docs/selects_doc.md#selectsconfig_setting_group)
macro in [Skylib](https://github.com/bazelbuild/bazel-skylib)'s
[`selects`](https://github.com/bazelbuild/bazel-skylib/blob/master/docs/selects_doc.md)
module supports `OR`ing multiple `config_setting`s:

```python
load("@bazel_skylib//lib:selects.bzl", "selects")
```


```python
config_setting(
    name = "config1",
    values = {"cpu": "arm"},
)
config_setting(
    name = "config2",
    values = {"compilation_mode": "dbg"},
)
selects.config_setting_group(
    name = "config1_or_2",
    match_any = [":config1", ":config2"],
)
sh_binary(
    name = "my_target",
    srcs = ["always_include.sh"],
    deps = select({
        ":config1_or_2": [":standard_lib"],
        "//conditions:default": [":other_lib"],
    }),
)
```

Unlike `selects.with_or`, different targets can share `:config1_or_2` across
different attributes.

It's an error for multiple conditions to match unless one is an unambiguous
"specialization" of the others. See [here](#example) for details.

## AND chaining

If you need a `select` branch to match when multiple conditions match, use the
[Skylib](https://github.com/bazelbuild/bazel-skylib) macro
[config_setting_group](https://github.com/bazelbuild/bazel-skylib/blob/master/docs/selects_doc.md#selectsconfig_setting_group):

```python
config_setting(
    name = "config1",
    values = {"cpu": "arm"},
)
config_setting(
    name = "config2",
    values = {"compilation_mode": "dbg"},
)
selects.config_setting_group(
    name = "config1_and_2",
    match_all = [":config1", ":config2"],
)
sh_binary(
    name = "my_target",
    srcs = ["always_include.sh"],
    deps = select({
        ":config1_and_2": [":standard_lib"],
        "//conditions:default": [":other_lib"],
    }),
)
```

Unlike OR chaining, existing `config_setting`s can't be directly `AND`ed
inside a `select`. You have to explicitly wrap them in a `config_setting_group`.

## Custom error messages

By default, when no condition matches, the target the `select()` is attached to
fails with the error:

```sh
ERROR: Configurable attribute "deps" doesn't match this configuration (would
a default condition help?).
Conditions checked:
  //tools/cc_target_os:darwin
  //tools/cc_target_os:android
```

This can be customized with the [`no_match_error`](be/functions.html#select)
attribute:

```python
cc_library(
    name = "my_lib",
    deps = select(
        {
            "//tools/cc_target_os:android": [":android_deps"],
            "//tools/cc_target_os:windows": [":windows_deps"],
        },
        no_match_error = "Please build with an Android or Windows toolchain",
    ),
)
```

```sh
$ bazel build //myapp:my_lib
ERROR: Configurable attribute "deps" doesn't match this configuration: Please
build with an Android or Windows toolchain
```

## <a name="rules"></a>Rules compatibility
Rule implementations receive the *resolved values* of configurable
attributes. For example, given:

```python
# myapp/BUILD

some_rule(
    name = "my_target",
    some_attr = select({
        ":foo_mode": [":foo"],
        ":bar_mode": [":bar"],
    }),
)
```

```sh
$ bazel build //myapp/my_target --define mode=foo
```

Rule implementation code sees `ctx.attr.some_attr` as `[":foo"]`.

Macros can accept `select()` clauses and pass them through to native
rules. But *they cannot directly manipulate them*. For example, there's no way
for a macro to convert

```python
select({"foo": "val"}, ...)
```

to

```python
select({"foo": "val_with_suffix"}, ...)
```

This is for two reasons.

First, macros that need to know which path a `select` will choose *cannot work*
because macros are evaluated in Bazel's [loading phase](guide.html#loading-phase),
which occurs before flag values are known.
This is a core Bazel design restriction that's unlikely to change any time soon.

Second, macros that just need to iterate over *all* `select` paths, while
technically feasible, lack a coherent UI. Further design is necessary to change
this.

## <a name="query"></a>Bazel query and cquery
Bazel [`query`](query-how-to.html) operates over Bazel's
[loading phase](https://docs.bazel.build/versions/master/glossary.html#loading-phase).
This means it doesn't know what command line flags a target uses since those
flags aren't evaluated until later in the build (in the
[analysis phase](https://docs.bazel.build/versions/master/glossary.html#analysis-phase)).
So it can't determine which `select()` branches are chosen.

Bazel [`cquery`](cquery.html) operates after Bazel's analysis phase, so it has
all this information and can accurately resolve `select()`s.

Consider:

```python
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
```
```python
# myapp/BUILD

string_flag(
    name = "dog_type",
    build_setting_default = "cat"
)

cc_library(
    name = "my_lib",
    deps = select({
        ":long": [":foo_dep"],
        ":short": [":bar_dep"],
    }),
)

config_setting(
    name = "long",
    flag_values = {":dog_type": "dachshund"},
)

config_setting(
    name = "short",
    flag_values = {":dog_type": "pug"},
)
```

`query` overapproximates `:my_lib`'s dependencies:

```sh
$ bazel query 'deps(//myapp:my_lib)'
//myapp:my_lib
//myapp:foo_dep
//myapp:bar_dep
```

while `cquery` shows its exact dependencies:

```sh
$ bazel cquery 'deps(//myapp:my_lib)' --//myapp:dog_type=pug
//myapp:my_lib
//myapp:bar_dep
```

## FAQ

### <a name="macros-select"></a>Why doesn't select() work in macros?
select() *does* work in rules! See [Rules compatibility](#rules) for
details.

The key issue this question usually means is that select() doesn't work in
*macros*. These are different than *rules*. See the
documentation on [rules](skylark/rules.html) and [macros](skylark/macros.html)
to understand the difference.
Here's an end-to-end example:

Define a rule and macro:

```python
# myapp/defs.bzl

# Rule implementation: when an attribute is read, all select()s have already
# been resolved. So it looks like a plain old attribute just like any other.
def _impl(ctx):
    name = ctx.attr.name
    allcaps = ctx.attr.my_config_string.upper()  # This works fine on all values.
    print("My name is " + name + " with custom message: " + allcaps)

# Rule declaration:
my_custom_bazel_rule = rule(
    implementation = _impl,
    attrs = {"my_config_string": attr.string()},
)

# Macro declaration:
def my_custom_bazel_macro(name, my_config_string):
    allcaps = my_config_string.upper()  # This line won't work with select(s).
    print("My name is " + name + " with custom message: " + allcaps)
```

Instantiate the rule and macro:

```python
# myapp/BUILD

load("//myapp:defs.bzl", "my_custom_bazel_rule")
load("//myapp:defs.bzl", "my_custom_bazel_macro")

my_custom_bazel_rule(
    name = "happy_rule",
    my_config_string = select({
        "//tools/target_cpu:x86": "first string",
        "//tools/target_cpu:ppc": "second string",
    }),
)

my_custom_bazel_macro(
    name = "happy_macro",
    my_config_string = "fixed string",
)

my_custom_bazel_macro(
    name = "sad_macro",
    my_config_string = select({
        "//tools/target_cpu:x86": "first string",
        "//tools/target_cpu:ppc": "other string",
    }),
)
```

Building fails because `sad_macro` can't process the `select()`:

```sh
$ bazel build //myapp:all
ERROR: /myworkspace/myapp/BUILD:17:1: Traceback
  (most recent call last):
File "/myworkspace/myapp/BUILD", line 17
my_custom_bazel_macro(name = "sad_macro", my_config_stri..."}))
File "/myworkspace/myapp/defs.bzl", line 4, in
  my_custom_bazel_macro
my_config_string.upper()
type 'select' has no method upper().
ERROR: error loading package 'myapp': Package 'myapp' contains errors.
```

Building succeeds when you comment out `sad_macro`:

```sh
# Comment out sad_macro so it doesn't mess up the build.
$ bazel build //myapp:all
DEBUG: /myworkspace/myapp/defs.bzl:5:3: My name is happy_macro with custom message: FIXED STRING.
DEBUG: /myworkspace/myapp/hi.bzl:15:3: My name is happy_rule with custom message: FIRST STRING.
```

This is impossible to change because *by definition* macros are evaluated before
Bazel reads the build's command line flags. That means there isn't enough
information to evaluate select()s.

Macros can, however, pass `select()`s as opaque blobs to rules:

```python
# myapp/defs.bzl

def my_custom_bazel_macro(name, my_config_string):
    print("Invoking macro " + name)
    my_custom_bazel_rule(
        name = name + "_as_target",
        my_config_string = my_config_string,
    )
```

```sh
$ bazel build //myapp:sad_macro_less_sad
DEBUG: /myworkspace/myapp/defs.bzl:23:3: Invoking macro sad_macro_less_sad.
DEBUG: /myworkspace/myapp/defs.bzl:15:3: My name is sad_macro_less_sad with custom message: FIRST STRING.
```

### <a name="boolean-select"></a>Why does select() always return true?
Because *macros* (but not rules) by definition
[can't evaluate `select()`s](#macros-select), any attempt to do so
usually produces an error:

```sh
ERROR: /myworkspace/myapp/BUILD:17:1: Traceback
  (most recent call last):
File "/myworkspace/myapp/BUILD", line 17
my_custom_bazel_macro(name = "sad_macro", my_config_stri..."}))
File "/myworkspace/myapp/defs.bzl", line 4, in
  my_custom_bazel_macro
my_config_string.upper()
type 'select' has no method upper().
```

Booleans are a special case that fail silently, so you should be particularly
vigilant with them:

```sh
$ cat myapp/defs.bzl
def my_boolean_macro(boolval):
  print("TRUE" if boolval else "FALSE")

$ cat myapp/BUILD
load("//myapp:defs.bzl", "my_boolean_macro")
my_boolean_macro(
    boolval = select({
        "//tools/target_cpu:x86": True,
        "//tools/target_cpu:ppc": False,
    }),
)

$ bazel build //myapp:all --cpu=x86
DEBUG: /myworkspace/myapp/defs.bzl:4:3: TRUE.
$ bazel build //mypro:all --cpu=ppc
DEBUG: /myworkspace/myapp/defs.bzl:4:3: TRUE.
```

This happens because macros don't understand the contents of `select()`.
So what they're really evaluting is the `select()` object itself. According to
[Pythonic](https://docs.python.org/release/2.5.2/lib/truth.html) design
standards, all objects aside from a very small number of exceptions
automatically return true.
### <a name="inspectable-select"></a>Can I read select() like a dict?
Macros [can't](#macros-select) evaluate select(s) because macros evaluate before
Bazel knows what the build's command line parameters are. Can they at least read
the `select()`'s dictionary to, for example, add a suffix to each value?

Conceptually this is possible, but it isn't yet a Bazel feature.
What you *can* do today is prepare a straight dictionary, then feed it into a
`select()`:

```sh
$ cat myapp/defs.bzl
def selecty_genrule(name, select_cmd):
  for key in select_cmd.keys():
    select_cmd[key] += " WITH SUFFIX"
  native.genrule(
      name = name,
      outs = [name + ".out"],
      srcs = [],
      cmd = "echo " + select(select_cmd + {"//conditions:default": "default"})
        + " > $@"
  )

$ cat myapp/BUILD
selecty_genrule(
    name = "selecty",
    select_cmd = {
        "//tools/target_cpu:x86": "x86 mode",
    },
)

$ bazel build //testapp:selecty --cpu=x86 && cat bazel-genfiles/testapp/selecty.out
x86 mode WITH SUFFIX
```

If you'd like to support both `select()` and native types, you can do this:

```sh
$ cat myapp/defs.bzl
def selecty_genrule(name, select_cmd):
    cmd_suffix = ""
    if type(select_cmd) == "string":
        cmd_suffix = select_cmd + " WITH SUFFIX"
    elif type(select_cmd) == "dict":
        for key in select_cmd.keys():
            select_cmd[key] += " WITH SUFFIX"
        cmd_suffix = select(select_cmd + {"//conditions:default": "default"})

    native.genrule(
        name = name,
        outs = [name + ".out"],
        srcs = [],
        cmd = "echo " + cmd_suffix + "> $@",
    )
```

### <a name="bind-select"></a>Why doesn't select() work with bind()?

Because [`bind()`](be/workspace.html#bind) is a WORKSPACE rule, not a BUILD rule.

Workspace rules do not have a specific configuration, and aren't evaluated in
the same way as BUILD rules. Therefore, a `select()` in a `bind()` can't
actually evaluate to any specific branch.

Instead, you should use [`alias()`](be/general.html#alias), with a `select()` in
the `actual` attribute, to perform this type of run-time determination. This
works correctly, since `alias()` is a BUILD rule, and is evaluated with a
specific configuration.

You can even have a `bind()` target point to an `alias()`, if needed.

```sh
$ cat WORKSPACE
workspace(name = "myapp")
bind(name = "openssl", actual = "//:ssl")
http_archive(name = "alternative", ...)
http_archive(name = "boringssl", ...)

$ cat BUILD
config_setting(
    name = "alt_ssl",
    define_values = {
        "ssl_library": "alternative",
    },
)

alias(
    name = "ssl",
    actual = select({
        "//:alt_ssl": "@alternative//:ssl",
        "//conditions:default": "@boringssl//:ssl",
    }),
)
```

With this setup, you can pass `--define ssl_library=alternative`, and any target
that depends on either `//:ssl` or `//external:ssl` will see the alternative
located at `@alternative//:ssl`.

### Why doesn't my select() choose what I expect?
If `//myapp:foo` has a `select()` that doesn't choose the condition you expect,
use [cquery](cquery.html) and `bazel config` to debug:

If `//myapp:foo` is the top-level target you're building, run:

```sh
$ bazel cquery //myapp:foo <desired build flags>
//myapp:foo (12e23b9a2b534a)
```

If you're building some other target `//bar` that depends on
//myapp:foo somewhere in its subgraph, run:

```sh
$ bazel cquery 'somepath(//bar, //myapp:foo)' <desired build flags>
//bar:bar   (3ag3193fee94a2)
//bar:intermediate_dep (12e23b9a2b534a)
//myapp:foo (12e23b9a2b534a)
```

The `(12e23b9a2b534a)` next to `//myapp:foo` is a *hash* of the
configuration that resolves `//myapp:foo`'s `select()`. You can inspect its
values with `bazel config`:

```sh
$ bazel config 12e23b9a2b534a
BuildConfiguration 12e23b9a2b534a
Fragment com.google.devtools.build.lib.analysis.config.CoreOptions {
  cpu: darwin
  compilation_mode: fastbuild
  ...
}
Fragment com.google.devtools.build.lib.rules.cpp.CppOptions {
  linkopt: [-Dfoo=bar]
  ...
}
...
```

Then compare this output against the settings expected by each `config_setting`.

`//myapp:foo` may exist in different configurations in the same build. See the
[cquery docs](cquery.html) for guidance on using `somepath` to get the right
one.

Caution: To prevent restarting the Bazel server, invoke `bazel config` with the
same command line flags as the `bazel cquery`. The `config` command relies on
the configuration nodes from the still-running server of the previous command.

[BuildSettings]: skylark/config.html#user-defined-build-settings

---
layout: documentation
title: Configurable Build Attributes
---

# Configurable Build Attributes

### Contents
* [Example](#example)
* [Configuration Conditions](#configuration-conditions)
* [Defaults](#defaults)
* [Custom Keys](#custom-keys)
* [Platforms](#platforms)
* [Short Keys](#short-keys)
* [Multiple Selects](#multiple-selects)
* [OR Chaining](#or-chaining)
* [Custom Error Messages](#custom-error-messages)
* [Rules Compatibility](#rules)
* [Bazel Query and Cquery](#query)
* [FAQ](#faq)
  * [Why doesn't select() work in macros?](#macros-select)
  * [Why does select() always return true?](#boolean-select)
  * [Can I read select() like a dict?](#inspectable-select)
  * [Why doesn't select() work with bind()?](#bind-select)

&nbsp;

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
*configuration conditions*. These conditions are labels that refer to
[`config_setting`](be/general.html#config_setting) targets. By using `select()`
in a configurable attribute, the attribute effectively takes on different values
when different conditions hold.

Matches must be unambiguous: either exactly one condition must match or, if
multiple conditions match, one's `values` must be a strict superset of all
others'. For example, `values = {"cpu": "x86", "compilation_mode": "dbg"}` is an
unambiguous specialization of `values = {"cpu": "x86"}`. The built-in condition
[`//conditions:default`](#defaults) automatically matches when nothing else
does.

This example uses `deps`. But `select()` works just as well on `srcs`,
`resources`, `cmd`, or practically any other attribute. Only a small number of
attributes are *non-configurable*, and those are clearly annotated; for
instance, `config_setting`'s own
[`values`](be/general.html#config_setting.values) attribute is non-configurable.

Certain attributes, like the `tools` of a `genrule`, have the effect of changing
the build parameters (such as the cpu) for all targets that transitively appear
beneath them. This will affect how conditions are matched within those targets
but not within the attribute that causes the change. That is, a `select` in the
`tools` attribute of a `genrule` will work the same as a `select` in the `srcs`.

## Configuration Conditions

Each key in a configurable attribute is a label reference to a
[`config_setting`](be/general.html#config_setting) target. This is just a
collection of expected command line flag settings. By encapsulating these in a
target, it's easy to maintain "standard" conditions that can be referenced
across targets and BUILD files.

The core `config_setting` syntax is:

```python
config_setting(
    name = "meaningful_condition_name",
    values = {
        "flag1": "expected_value1",
        "flag2": "expected_value2",
        ...
    },
)
```

`flagN` is an arbitrary Bazel command line flag. `value` is the expected value
for that flag. A `config_setting` matches when *all* of its flags match (order
is irrelevant).

`values` entries use the same parsing logic as at the actual command line. This
means:

*  `values = { "compilation_mode": "opt" }` matches `bazel build -c opt ...`
*  `values = { "java_header_compilation": "true" }` matches `bazel build
--java_header_compilation=1 ...`
*  `values = { "java_header_compilation": "0" }` matches `bazel build
--nojava_header_compilation ...`

`config_setting` only works with flags that affect build rule output. For
example, [`--show_progress`](user-manual.html#flag--show_progress) isn't allowed
because this only affects how Bazel reports progress to the user.

`config_setting` semantics are intentionally simple. For example, there's no
direct support for `OR` chaining (although a
[convenience function](#or-chaining) provides this).  Consider writing
macros for complicated flag logic.

## Defaults

The built-in condition `//conditions:default` matches when no other condition
matches.

Because of the "exactly one match" rule, a configurable attribute with no match
and no default condition triggers a `"no matching conditions"` error. This can
protect against silent failures from unexpected build flags:

```python
# foo/BUILD

config_setting(
    name = "foobar",
    values = {"define": "foo=bar"},
)

cc_library(
    name = "my_lib",
    srcs = select({
        ":foobar": ["foobar_lib.cc"],
    }),
)
```

```sh
$ bazel build //foo:my_lib --define foo=baz
ERROR: Configurable attribute "srcs" doesn't match this configuration (would
a default condition help?).
Conditions checked:
  //foo:foobar
```

`select()` can include a [`no_match_error`](#custom-error-messages) for custom
failure messages.

## Custom Keys

Since `config_setting` currently only supports built-in Bazel flags, the level
of custom conditioning it can support is limited. For example, there's no Bazel
flag for `IncludeSpecialProjectFeatureX`.

Plans for [truly custom flags](
https://docs.google.com/document/d/1vc8v-kXjvgZOdQdnxPTaV0rrLxtP2XwnD2tAZlYJOqw/edit?usp=sharing)
are underway. In the meantime, [`--define`](user-manual.html#flag--define) is
the best approach for these purposes.
`--define` is a bit awkward to use and wasn't originally designed for this
purpose. We recommend using it sparingly until true custom flags are available.
For example, don't use `--define` to specify multiple variants of top-level
binary. Just use multiple targets instead.

To trigger an arbitrary condition with `--define`, write

```python
config_setting(
    name = "bar",
    values = {"define": "foo=bar"},
)

config_setting(
    name = "baz",
    values = {"define": "foo=baz"},
)
```

and run `$ bazel build //my:target --define foo=baz`.

The `values` attribute can't contain multiple `define`s. This is
because each instance has the same dictionary key. To solve this, use
`define_values`:

```python
config_setting(
    name = "bar_and_baz",
    define_values = {
        "foo": "bar",  # matches --define foo=bar
        "baz": "bat",  # matches --define baz=bat
    },
)
```

When `define`s appear in both `values` and `define_values`, all must match for
the `config_setting` to match.

## Platforms

While the ability to specify multiple flags on the command line provides
flexibility, it can also be burdensome to individually set each one every time
you want to build a target.
   [Platforms](platforms.html)
allow you to consolidate these into simple bundles.

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
we can simply run

```sh
bazel build //my_app:my_rocks --platforms=//myapp:marble_platform
```

Without platforms, this might look something like

```sh
bazel build //my_app:my_rocks --define color=white --define texture=smooth --define type=metamorphic
```

Platforms are still under development. See the [documentation](platforms.html)
and [roadmap](https://bazel.build/roadmaps/platforms.html) for details.

## Short Keys

Since configuration keys are target labels, their names can get long and
unwieldy. This can be mitigated with local variable definitions:

Before:

```python
sh_binary(
    name = "my_target",
    srcs = select({
        "//my/project/my/team/configs:config1": ["my_target_1.sh"],
        "//my/project/my/team/configs:config2": ["my_target_2.sh"],
    }),
)
```

After:

```python
CONFIG1="//my/project/my/team/configs:config1"
CONFIG2="//my/project/my/team/configs:config2"

sh_binary(
    name = "my_target",
    srcs = select({
        CONFIG1: ["my_target_1.sh"],
        CONFIG2: ["my_target_2.sh"],
    })
)
```


For more complex expressions, you can use [macros](skylark/macros.md):

Before:

```python
# foo/BUILD

genrule(
    name = "my_target",
    srcs = [],
    outs = ["my_target.out"],
    cmd = select({
        "//my/project/my/team/configs/config1": "echo custom val: this > $@",
        "//my/project/my/team/configs/config2": "echo custom val: that > $@",
        "//conditions:default": "echo default output > $@",
    }),
)
```

After:

```python
# foo/genrule_select.bzl

def select_echo(input_dict):
    echo_cmd = "echo %s > $@"
    out_dict = {"//conditions:default": echo_cmd % "default output"}
    for (key, val) in input_dict.items():
        cmd = echo_cmd % ("custom val: " + val)
        out_dict["//my/project/my/team/configs/config" + key] = cmd
    return select(out_dict)
```

```python
# foo/BUILD

load("//foo:genrule_select.bzl", "select_echo")

genrule(
    name = "my_target",
    srcs = [],
    outs = ["my_target.out"],
    cmd = select_echo({
        "1": "this",
        "2": "that",
    }),
)
```

## Multiple Selects

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

`select` cannot appear inside another `select` (i.e. *`AND` chaining*). If you
need to `AND` selects together, either use an intermediate target:

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

or write a [macro](skylark/macros.md) to do the same thing
automatically.

This approach doesn't work for non-deps attributes (like
[genrule:cmd](be/general.html#genrule.cmd)). For these, extra `config_settings`
may be necessary:

```python
config_setting(
    name = "armeabi_and_opt",
    values = {
        "cpu": "armeabi",
        "compilation_mode": "opt",
    },
)
```


## OR Chaining

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

Most conditions evaluate to the same dep. But this syntax is verbose, hard to
maintain, and refactoring-unfriendly. It would be nice to not have to repeat
`[":standard_lib"]` over and over.

One option is to predefine the declaration as a BUILD variable:

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

This makes it easier to manage the dependency. But it still adds unnecessary
duplication.

`select()` doesn't support native syntax for `OR`ed conditions. For this, use
the [Skylib](https://github.com/bazelbuild/bazel-skylib) utility [`selects`](
https://github.com/bazelbuild/bazel-skylib/blob/master/lib/selects.bzl).

```python
load("@bazel_skylib//:lib.bzl", "selects")
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

This automatically expands the `select` to the original syntax above.

For `AND` chaining, see [here](#multiple-selects).

## Custom Error Messages

By default, when no condition matches, the owning target fails with the error:

```sh
ERROR: Configurable attribute "deps" doesn't match this configuration (would
a default condition help?).
Conditions checked:
  //tools/cc_target_os:darwin
  //tools/cc_target_os:android
```

This can be customized with [`no_match_error`](be/functions.html#select):

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
$ bazel build //foo:my_lib
ERROR: Configurable attribute "deps" doesn't match this configuration: Please
build with an Android or Windows toolchain
```

## <a name="rules"></a>Rules Compatibility
Rule implementations receive the *resolved values* of configurable
attributes. For example, given:

```python
# myproject/BUILD

some_rule(
    name = "my_target",
    some_attr = select({
        ":foo_mode": [":foo"],
        ":bar_mode": [":bar"],
    }),
)
```

```sh
$ bazel build //myproject/my_target --define mode=foo
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
because macros are evaluated in Bazel's [loading phase](user-manual.html#loading-phase),
which occurs before flag values are known.
This is a core Bazel design restriction that's unlikely to change any time soon.

Second, macros that just need to iterate over *all* `select` paths, while
technically feasible, lack a coherent UI. Further design is necessary to change
this.

## <a name="query"></a>Bazel Query and Cquery
Bazel `query` operates over Bazel's [loading phase](
user-manual.html#loading-phase). This means it doesn't know what command line
flags will be applied to a target since those flags aren't evaluated until later
in the build (during the [analysis phase](user-manual.html#analysis-phase)). So
the [`query`](query.html) command can't accurately determine which path a
configurable attribute will follow.

[Bazel `cquery`](cquery.html) has the advantage of being able to parse build
flags and operating post-analysis phase so it correctly resolves configurable
attributes. It doesn't have full feature parity with query but supports most
major functionality and is actively being worked on.
Querying the following build file...

```python
# myproject/BUILD

cc_library(
    name = "my_lib",
    deps = select({
        ":long": [":foo_dep"],
        ":short": [":bar_dep"],
    }),
)

config_setting(
    name = "long",
    values = {"define": "dog=dachshund"},
)

config_setting(
    name = "short",
    values = {"define": "dog=pug"},
)
```

...would return the following results.

```sh
$ bazel query 'deps(//myproject:my_lib)'
//myproject:my_lib
//myproject:foo_dep
//myproject:bar_dep

$ bazel cquery 'deps(//myproject:my_lib)' --define dog=pug
//myproject:my_lib
//myproject:bar_dep
```

## FAQ

## <a name="macros-select"></a>Why doesn't select() work in macros?
select() *does* work in rules! See [Rules compatibility](#rules) for
details.

The key issue this question usually means is that select() doesn't work in
*macros*. These are different than *rules*. See the
documentation on [rules](skylark/rules.html) and [macros](skylark/macros.html)
to understand the difference.
Here's an end-to-end example:

Define a rule and macro:

```python
# myproject/defs.bzl

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
# myproject/BUILD

load("//myproject:defs.bzl", "my_custom_bazel_rule")
load("//myproject:defs.bzl", "my_custom_bazel_macro")

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
$ bazel build //myproject:all
ERROR: /myworkspace/myproject/BUILD:17:1: Traceback
  (most recent call last):
File "/myworkspace/myproject/BUILD", line 17
my_custom_bazel_macro(name = "sad_macro", my_config_stri..."}))
File "/myworkspace/myproject/defs.bzl", line 4, in
  my_custom_bazel_macro
my_config_string.upper()
type 'select' has no method upper().
ERROR: error loading package 'myproject': Package 'myproject' contains errors.
```

Building succeeds when we comment out `sad_macro`:

```sh
# Comment out sad_macro so it doesn't mess up the build.
$ bazel build //myproject:all
DEBUG: /myworkspace/myproject/defs.bzl:5:3: My name is happy_macro with custom message: FIXED STRING.
DEBUG: /myworkspace/myproject/hi.bzl:15:3: My name is happy_rule with custom message: FIRST STRING.
```

This is impossible to change because *by definition* macros are evaluated before
Bazel reads the build's command line flags. That means there isn't enough
information to evaluate select()s.

Macros can, however, pass `select()`s as opaque blobs to rules:

```python
# myproject/defs.bzl

def my_custom_bazel_macro(name, my_config_string):
    print("Invoking macro " + name)
    my_custom_bazel_rule(
        name = name + "_as_target",
        my_config_string = my_config_string,
    )
```

```sh
$ bazel build //myproject:sad_macro_less_sad
DEBUG: /myworkspace/myproject/defs.bzl:23:3: Invoking macro sad_macro_less_sad.
DEBUG: /myworkspace/myproject/defs.bzl:15:3: My name is sad_macro_less_sad with custom message: FIRST STRING.
```

## <a name="boolean-select"></a>Why does select() always return true?
Because *macros* (but not rules) by definition
[can't evaluate select(s)](#macros-select), any attempt to do so
usually produces a an error:

```sh
ERROR: /myworkspace/myproject/BUILD:17:1: Traceback
  (most recent call last):
File "/myworkspace/myproject/BUILD", line 17
my_custom_bazel_macro(name = "sad_macro", my_config_stri..."}))
File "/myworkspace/myproject/defs.bzl", line 4, in
  my_custom_bazel_macro
my_config_string.upper()
type 'select' has no method upper().
```

Booleans are a special case that fail silently, so you should be particularly
vigilant with them:

```sh
$ cat myproject/defs.bzl
def my_boolean_macro(boolval):
  print("TRUE" if boolval else "FALSE")

$ cat myproject/BUILD
load("//myproject:defs.bzl", "my_boolean_macro")
my_boolean_macro(
    boolval = select({
        "//tools/target_cpu:x86": True,
        "//tools/target_cpu:ppc": False,
    }),
)

$ bazel build //myproject:all --cpu=x86
DEBUG: /myworkspace/myproject/defs.bzl:4:3: TRUE.
$ bazel build //myproject:all --cpu=ppc
DEBUG: /myworkspace/myproject/defs.bzl:4:3: TRUE.
```

This happens because macros don't understand the contents of `select()`.
So what they're really evaluting is the `select()` object itself. According to
[Pythonic](https://docs.python.org/release/2.5.2/lib/truth.html) design
standards, all objects aside from a very small number of exceptions
automatically return true.
## <a name="inspectable-select"></a>Can I read select() like a dict?
Fine. Macros [can't](#macros-select) evaluate select(s) because
macros are evaluated before Bazel knows what the command line flags are.

Can macros at least read the `select()`'s dictionary, say, to add an extra
suffix to each branch?

Conceptually this is possible. But this isn't yet implemented and is not
currently prioritized.
What you *can* do today is prepare a straight dictionary, then feed it into a
`select()`:

```sh
$ cat myproject/defs.bzl
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

$ cat myproject/BUILD
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
$ cat myproject/defs.bzl
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

## <a name="bind-select"></a>Why doesn't select() work with bind()?
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
workspace(name = "myproject")
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
that depends on either `//:ssl` or `//external:openssl` will see the alternative
located at `@alternative//:ssl`.

# Configurable Build Attributes

go/configurable-build-attributes

[TOC]

**Configurable attributes** is a Blaze feature that lets BUILD rule attributes
determine their values from command-line flags.

This can be used, for example, to declare custom deps controllable at the
command line.


## Example

```
//myapp/BUILD:
sh_binary(
    name = "my_rule",
    srcs = select({
        ":armeabi_mode": ["armeabi_src.sh"],
        ":x86_opt_mode": ["x86_opt_src.sh"],
        "//conditions:default": ["generic_src.sh"]
    })
)

config_setting(
    name = "armeabi_mode",
    values = { "cpu": "armeabi" }
)

config_setting(
    name = "x86_opt_mode",
    values = {
      "cpu": "x86",
      "compilation_mode" : "opt",
    }
)
```

This declares an `sh_binary` that "chooses" its source based on the flags at the
command line. Specficially, `srcs` becomes:


Command | `srcs =`
------- | --------
`blaze build //myapp:my_rule --cpu=armeabi` | `["armeabi_src.sh"]`
`blaze build //myapp:my_rule --c opt --cpu=x86` | `["x86_opt_src.sh"]`
`blaze build //myapp:my_rule` | `["generic_src.sh"]`
`blaze build //myapp:my_rule -c opt` | `["generic_src.sh"]`

In essence, `select()` turns any attribute into a dictionary that maps
configuration conditions to desired values. Configuration conditions are label
references to `config_setting` rules. Values are any valid value the attribute
can normally take. The match must be unambiguous: either exactly one condition
must match, or else one condition must be a refinement of all other matches. The
built-in condition `//conditions:default` automatically matches when nothing
else matches.

This example uses `srcs`, but this works just as well for `deps`, `resources`,
`proguard_specs`, `cmd`, or almost any other attribute. Only a small number of
attributes are *non-configurable*;
see [here](http://go/nonconfigurable-blaze-attributes) for the complete list.

## Configuration Conditions

Each key in a configurable attribute is a label reference to a
[`config_setting`](be/general.html#config_setting) rule. This provides a
*named*, *structured* reference point for expected conditions and the ability to
define the exact flags that fulfill them. This facilitates *domain-wide*
definitions that can be shared across multiple rules and BUILD files.

The core `config_setting` syntax is:

```
config_setting(
    name = "meaningful_condition_name",
    values = {
        "flag1": "value1",
        "flag2": "value2",
        ...
    }
)
```

`flagN` is an arbitrary Blaze command-line flag (most flags are supported, but
[not all](http://goto.corp.google.com/configurable-blaze-flags)). `value` is the
expected value for that flag. A `config_setting` matches a build when *all* of
its flag expectations match.

`values` entries use the same parsing logic as command-line flags This means:

*  `values = { "compilation_mode": "opt" }` matches `blaze build -c opt ...`
*  `values = { "java_header_compilation": "true" }` matches `blaze build
--java_header_compilation=1 ...`
*  `values = { "translations": "0" }` matches `blaze build --notranslations
   ...`

If you want to trigger on generic conditions like *Android* or *Windows*, use
a [predefined condition](#predefined-conditions). As a general principle it's
better to have one `config_setting` shared across projects vs. redefining the
same setting over and over again.

`config_setting` semantics are intentionally simple. For example, there is no
direct support for `OR` chaining (although a
[Skylark convenience function](#or-chaining) provides this).  Consider writing
Skylark macros for complicated flag logic.

## Defaults

The built-in condition `//conditions:default` matches any configurable attribute
when nothing else matches.

Because of the "exactly one match" rule, a configurable attribute with no match
and no default condition triggers a `"no matching conditions"` build error. This
can protect against silent failures like `blaze build --define
foo=oopsIDidntMeanThis`:

```
//foo:
config_setting(
    name = "foobar",
    values = { "define": "foo=bar" }
)

cc_library(
    name = "my_lib",
    srcs = select({
        ":foobar": ["foobar_lib.cc"],
    })
)
```

```
$ blaze build //foo:my_lib --define foo=baz
ERROR: Configurable attribute "srcs" doesn't match this configuration (would
a default condition help?).
Conditions checked:
  //foo:foobar
```

## Custom Keys {#custom-keys}

Since `config_setting` only supports Blaze-recognized flags, this limits support
for conditions that Blaze doesn't natively understand. This can be a problem for
users who want to trigger on app-specific criteria like
`IncludeSpecialAppFeatureX`

Plans for [first-class custom
flags](http://go/skylark-build-configuration) are in the pipeline, but not ready
yet. In the meantime,
[`--define`](http://goto.corp.google.com/bum#flag--define) is the reluctantly
endorsed method for specifiying arbitrary criteria. `--define` is imperfect and
will be deprecated once proper custom flags are ready. Therefore, only use it
when there's no viable alternative. For example, don't use `--define` to specify
a handful of variants of top-level targets; just create multiple targets
instead.

To trigger an arbitrary value with `--define`, write:

```
config_setting(
    name = "bar",
    values = { "define": "foo=bar" }
)

config_setting(
    name = "baz",
    values = { "define": "foo=baz" }
)
```

and invoke the desired condition via `blaze build //my:target --define foo=baz`.

`values` cannot contain multiple `define`s. This is because the BUILD language
doesn't allow duplicate keys in a dictionary. To solve this, use
`define_values`:

```
config_setting(
    name = "bar_and_baz",
    define_values = {
        "foo": "bar", # matches --define foo=bar
        "baz": "bat", # matches --define baz=bat
    }
)
```

When `define`s appear in both `values` and `define_values`, all must match for
the `config_setting` to match.

## Platforms

While the ability to specify multiple flags on the command line provides
flexibility, it can also be burdensome to individually set each flag every
time you want to build a target. The ['--experimental_platforms'] (https://g3doc.corp.google.com/devtools/blaze/g3doc/be/platform.html#constraint_setting)
flag allows you to define a platform and use that as a basis for select().

```
sh_binary(
    name = "my_rocks_rule",
    srcs = select({
        ":basalt" : ["pyroxene.sh"],
        ":marble" : ["calcite.sh"],
        "//conditions:default": ["feldspar.sh"]
    })
)

config_setting(
    name = "basalt",
    constraint_values = [
        ":igneous",
        ":black"
    ]
)

config_setting(
    name = "marble",
    constraint_values = [
        ":white",
        ":metamorphic"
        ":smooth"
    ]
)

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
        ":igneous"]
)

platform(
    name = "marble_platform",
    constraint_values = [
        ":white",
        ":smooth"
        ":metamorphic"
    ]
)
```

The `platform` specified on the command line matches a `config_setting` that
contains the same set (or a superset) of `constraint_values` and triggers
that `config_setting` as a match in the `select()` statement.

For example, in order to set the `srcs` attribute of `my_rocks_rule` to
`calcite.sh`, simply run `blaze build my_app:my_rock_rule
--experimental_platforms=marble_platform` instead of `blaze build my_app:my_rule
--color=light --texture=smooth --define type=metamorphic`.

## Short Keys

Since configuration keys are rule labels, they have the potential to get
unwieldy. This can be mitigated through local variable definitions:

*Before:*

```
sh_binary(
    name = "my_rule",
    srcs = select({
        "//my/project/my/team/configs:config1": ["my_rule_1.sh"],
        "//my/project/my/team/configs:config2": ["my_rule_2.sh"],
    })
)
```

*After:*

```
CONFIG1="//my/project/my/team/configs:config1"
CONFIG2="//my/project/my/team/configs:config2"

sh_binary(
    name = "my_rule",
    srcs = select({
        CONFIG1: ["my_rule_1.sh"],
        CONFIG2: ["my_rule_2.sh"],
    })
)
```

For more complex
expressions, [Skylark macros](/devtools/blaze/rules/g3doc/macros.md) can
encapsulate required logic in easier to read form:

*Before:*

```
//foo/BUILD
genrule(
    name = "my_rule",
    srcs = [],
    outs = ["my_rule.out"],
    cmd = select({
        "//my/project/my/team/configs/config1": "echo custom val: this > $@",
        "//my/project/my/team/configs/config2": "echo custom val: that > $@",
        "//conditions:default": "echo default output > $@"
    })
)
```

*After:*

```
//foo/genrule_select.bzl:
def select_echo(input_dict):
  echo_cmd = "echo %s > $@"
  out_dict = {"//conditions:default": echo_cmd % "default output" }
  for (key, val) in input_dict.items():
    cmd = echo_cmd % ("custom val: " + val)
    out_dict["//my/project/my/team/configs/config" + key] = cmd
  return select(out_dict)
```

```
//foo/BUILD:
load("//foo:genrule_select.bzl", "select_echo")
genrule(
    name = "my_rule",
    srcs = [],
    outs = ["my_rule.out"],
    cmd = select_echo({
        "1": "this",
        "2": "that",
    })
)
```

## Multiple Selects {#multiple-selects}

`select` can appear multiple times in the same attribute assignment:

```
sh_binary(
    name = "my_rule",
    srcs = ["always_include.sh"]
    + select({
          ":armeabi_mode": ["armeabi_src.sh"],
          ":k8_mode": ["k8_src.sh"],
      })
    + select({
          ":opt_mode": ["opt_extras.sh"],
          ":dbg_mode": ["dbg_extras.sh"],
      })
)
```

`select` cannot appear inside another `select` (i.e. *`AND` chaining*). If you
need to `AND` selects together, either defer through an intermediate rule:

```
sh_binary
    name = "my_rule",
    srcs = ["always_include.sh"],
    deps = select({
        ":armeabi_mode": [":armeabi_lib"],
        ...
    })
)

sh_library(
    name = "armeabi_lib",
    srcs = select({
        ":opt_mode":  ["armeabi_with_opt.sh"],
        ...
    })
)
```

or write a [Skylark macro](/devtools/blaze/rules/g3doc/macros.md) that
automatically creates the intermediate rule for the user.

This approach doesn't work for non-deps attributes (e.g.
[genrule.cmd](http://go/be#genrule.cmd)). In that case extra `config_settings`
may be necessary:

```
config_setting(
    name = "armeabi_and_opt",
    values = {
        "cpu": "armeabi",
        "compilation_mode": "opt"
    }
)
```


## OR Chaining {#or-chaining}

Consider the following:

```
sh_binary
    name = "my_rule",
    srcs = ["always_include.sh"],
    deps = select({
        ":config1": [":standard_lib"],
        ":config2": [":standard_lib"],
        ":config3": [":standard_lib"],
        ":config4": [":special_lib"],
    })
)
```

Most conditions evaluate to the same dep. But this syntax is verbose,
burdensome to maintain, and refactoring-unfriendly. It would be nice not to
have to repeat `[":standard_lib"]` multiple times.

One option is to predefine the declaration as a BUILD variable:

```python
STANDARD_DEP = [":standard_lib"]

sh_binary
    name = "my_rule",
    srcs = ["always_include.sh"],
    deps = select({
        ":config1": STANDARD_DEP,
        ":config2": STANDARD_DEP,
        ":config3": STANDARD_DEP,
        ":config4": [":special_lib"],
    })
)
```

This makes it easier to manage the standard dependency. But it still requires
unnecessary duplication.

`select()` does not support native syntax for `OR`ed conditions. For this, use
the [Skylib](https://g3doc.corp.google.com/tools/build_defs/lib/README.md)
utility [selects](https://cs.corp.google.com/piper///depot/google3/tools/build_defs/lib/selects.bzl).
This adds support for the following syntax:

```
load("//tools/build_defs/lib:selects.bzl", "selects")

sh_binary
    name = "my_rule",
    srcs = ["always_include.sh"],
    deps = selects.with_or({
        (":config1", ":config2", ":config3"): [":standard_lib"],
        ":config4": [":special_lib"],
    })
)

```

This automatically expands the `select` to the original syntax shown above.

For `AND` chaining, see [here](#multiple-selects).

## Predefined Conditions {#predefined-conditions}

The following `config_setting`s have been predefined to support common
conditions likely to be shared across many teams. Use these whenever possible
instead of rolling your own:

*   [tools/cc_target_os/BUILD](http://cs.corp.google.com/#piper///depot/google3/tools/cc_target_os/BUILD)
    for "platform" criteria like *iOS* or *Android*

    *   If you need to combine platform information with another config_setting,
        use "cc_target_os" and not a hardcoded value for "crosstool_top".
        cc_target_os allows users to change the crosstool_top location with
        Blaze but still activate all the right conditions.

*   [tools/target_cpu/BUILD](https://cs.corp.google.com/#piper///depot/google3/tools/target_cpu/BUILD)
    for `--cpu` settings like `k8` or `arm`

*   [tools/compilation_mode/BUILD](https://cs.corp.google.com/#piper///depot/google3/tools/compilation_mode/BUILD)
    for `--compilation_mode` (`-c`) settings: `fastbuild`, `opt` or `dbg`. See
    g3doc/devtools/blaze/g3doc/user-manual.html#flag--compilation_mode

*   [tools/android_cpu/BUILD](https://cs.corp.google.com/#piper///depot/google3/tools/android_cpu/BUILD) -
    for `--android_cpu` settings like `arm64-v8a` or `x86`

Example:

```
cc_library(
    name = "my_lib",
    srcs = ["my_lib.cc"],
    deps = select({
        "//tools/cc_target_os:android": [":android_deps"],
        "//tools/cc_target_os:darwin": [":ios_deps"],
        "//tools/cc_target_os:windows": [":windows_deps"],
        "//conditions:default": [":linux_deps"],
    })
)
```

## Custom Error Messages

By default, when no condition matches, the owning rule fails with the error:

```
ERROR: Configurable attribute "deps" doesn't match this configuration (would
a default condition help?).
Conditions checked:
  //tools/cc_target_os:darwin
  //tools/cc_target_os:android
```

This can be customized via `no_match_error`:

```
cc_library(
    name = "my_lib",
    deps = select({
        "//tools/cc_target_os:android": [":android_deps"],
        "//tools/cc_target_os:windows": [":windows_deps"],
    }, no_match_error = "Please build with an Android or Windows toolchain"
    )
)
```

```
$ blaze build //foo:my_lib
ERROR: Configurable attribute "deps" doesn't match this configuration: Please
build with an Android or Windows toolchain
```

## Skylark Compatibility {#skylark}

Skylark is compatible with configurable attributes in limited form.

Skylark rule implementations receive the post-resolved outputs of configurable
attributes. For example, given:

```
//myproject/BUILD:
some_skylark_rule(
    name = "my_rule",
    some_attr = select({
        ":foo_mode": [":foo"],
        ":bar_mode": [":bar"],
    })
)
```

```
$ blaze build //myproject/my_rule --define mode=foo
```

Skylark rule implementation code sees `ctx.attr.some_attr` as `[":foo"]`.

Skylark macros can accept `select()` clauses and pass them through to native
rules. But *they cannot directly manipulate them*. For example, there's no way
for a Skylark macro to convert

```
`select({"foo": "val"}, ...)`
```

to

```
`select({"foo": "val_with_suffix"}, ...)`.
```

This is for two reasons.

First, macros that need to know which path a `select` will choose *cannot work*
because macros are evaluated in Blaze's *loading phase*, which occurs before
flag values are known. This is a core Blaze design restriction that's unlikely
to change any time soon.

Second, macros that just need to iterate over all `select` paths, while
technically feasible, lack a coherent
UI. See [b/23527731](https://b.corp.google.com/issues/23527731) for ongoing
discussion and proposals.

## Blaze Query and Cquery {#query}

Blaze `query` operates over Blaze's loading phase. This means it doesn't know
what command-line flags will be applied to a rule since those flags aren't
evaluated until later in the build (during the analysis phase). So the `query`
command cannot accurately determine which path a configurable attribute will
follow.

[Blaze `cquery`](https://g3doc.corp.google.com/devtools/blaze/subteams/configurability/g3doc/query.md)
has the advantage of being able to parse build flags and operating post-analysis
phase so it correctly resolves configurable attributes. It doesn't have full
feature parity with query but supports most major functionality and is actively
being worked on.

Querying the following build file...

```
//myproject/BUILD:
cc_library(
    name = "my_lib",
    deps = select({
        ":long": [":foo_dep"],
        ":short": [":bar_dep"],
    })
)
config_setting(
    name = 'long',
    values = { "define": "dog=dachshund" }
)
config_setting(
    name = 'short',
    values = { "define": "dog=pug" }
)
```
...would return the following results.

```
$ blaze query 'deps(//myproject:my_lib)'
//myproject:my_lib
//myproject:foo_dep
//myproject:bar_dep

$ blaze cquery 'deps(//myproject:my_lib)' --define dog=pug
//myproject:my_lib
//myproject:bar_dep
```

Since TAP relies on `query` to determine which builds are affected by a given
change, this means TAP can overschedule tests as a result.`Cquery` is not quite
ready to take this job over but is actively being improved. See [b/68317885]
(https://b.corp.google.com/issues/68317885) for updates and feel free to file
feature requests against the [`cquery` bug hotlist]
(https://buganizer.corp.google.com/hotlists/842234).

## Best Practices

In summary, consider the following when using configurable attributes:

1.  Consolidate `config_setting` definitions into common, shareable
    locations. Use [predefined conditions](#predefined-conditions)
    positions when possible. [Details](#predefined-conditions).
1.  Trigger your conditions on explicit Blaze flags whenever possible. If you
    absolutely must trigger on a custom setting that built-in Blaze flags can't
    model, use `--define`. This is an imperfect, stopgap measure until proper
    first-class custom flags are available. [Details](#custom-keys).
1.  Configurable attributes cause TAP and blaze `query` to overstate build
    dependencies. So make sure your rules only include conditions you actually
    use. For example, drop your Android dependencies if you know the Android
    version of your library is no longer used. This stops low-level Android
    changes from TAP-triggering Linux binaries. Or use
    blaze `cquery` for properly resolved configurable attributes but a smaller
    feature set.  [Details](#query).

## FAQ {#faq}

### Why doesn't select() work in Skylark? {#skylark-macros-select}
go/skylark-macros-select

select() *does* work in Skylark! See [Skylark compatibility](#skylark) for
details.

The key issue this question usually means is that select() doesn't work in
Skylark *macros*. These are different than Skylark *rules*. See the Skylark
documentation on [rules](https://g3doc.corp.google.com/devtools/blaze/rules/g3doc/rules.md)
and [macros](https://g3doc.corp.google.com/devtools/blaze/rules/g3doc/macros.md)
to understand the difference.

Here's an end-to-end example:

```
# myproject/defs.bzl:

# Rule implementation: when an attribute is read, all select()s have already
# been resolved. So it looks like a plain old attribute just like any other.
def _impl(ctx):
  name = ctx.attr.name
  allcaps = ctx.attr.my_config_string.upper()  # This works fine on all values.
  print("My name is " + name + " with custom message: " + allcaps)

# Skylark rule declaration:
my_custom_blaze_rule = rule(
    implementation = _impl,
    attrs = {"my_config_string": attr.string()}
)

# Skylark macro declaration:
def my_custom_blaze_macro(name, my_config_string):
  allcaps = my_config_string.upper() # This line won't work with select(s).
  print("My name is " + name + " with custom message: " + allcaps)


```

```
# myproject/BUILD:
load("//myproject:defx.bzl", "my_custom_blaze_rule")
load("//myproject:defs.bzl", "my_custom_blaze_macro")

my_custom_blaze_rule(
    name = "happy_rule",
    my_config_string = select({
        "//tools/target_cpu:k8": "first string",
        "//tools/target_cpu:ppc": "second string",
    }),
)

my_custom_blaze_macro(
    name = "happy_macro",
    my_config_string = "fixed string",
)

my_custom_blaze_macro(
    name = "sad_macro",
    my_config_string = select({
        "//tools/target_cpu:k8": "first string",
        "//tools/target_cpu:ppc": "other string",
    }),
)
```

```sh
$ blaze build //myproject:all
ERROR: /google/src/cloud/user/select/google3/myproject/BUILD:17:1: Traceback
  (most recent call last):
File "/google/src/cloud/user/select/google3/myproject/BUILD", line 17
my_custom_blaze_macro(name = "sad_macro", my_config_stri..."}))
File "/google/src/cloud/user/select/google3/myproject/defs.bzl", line 4, in
  my_custom_blaze_macro
my_config_string.upper()
type 'select' has no method upper().
ERROR: error loading package 'myproject': Package 'myproject' contains errors.
```

```sh
# Comment out sad_macro so it doesn't mess up the build.
$ blaze build //myproject:all
DEBUG: /google/src/cloud/user/select/google3/myproject/defs.bzl:5:3: My name is
  happy_macro with custom message: FIXED STRING.
DEBUG: /google/src/cloud/user/select/google3/myproject/hi.bzl:15:3: My name is
  happy_rule with custom message: FIRST STRING.
```

This is impossible to change because *by definition* macros are evaluated before
Blaze reads the build's command line flags. That means there isn't enough
information to evaluate select()s.

Macros can, however, pass select()s as opaque blobs to rules:

```
# myproject/defs.bzl:
def my_custom_blaze_macro(name, my_config_string):
  print("Invoking macro " + name)
  my_custom_blaze_rule(
      name = name + "_as_rule",
      my_config_string = my_config_string)
```

```sh
$ blaze build //myproject:sad_macro_less_sad
DEBUG: /google/src/cloud/user/select/google3/testapp/defs.bzl:23:3:
  Invoking macro sad_macro.
DEBUG: /google/src/cloud/uer/select/google3/testapp/defs.bzl:15:3: My name is
  sad_macro_less_sad with custom message: FIRST STRING.
```

### Why does select() always return true in Skylark? {#boolean-select}
go/boolean-select

Because Skylark *macros* (but not rules) by definition
[can't evaluate select(s)](#skylark-macros-select), any attempt to do so
usually produces a an error:

```sh
ERROR: /google/src/cloud/user/select/google3/myproject/BUILD:17:1: Traceback
  (most recent call last):
File "/google/src/cloud/user/select/google3/myproject/BUILD", line 17
my_custom_blaze_macro(name = "sad_macro", my_config_stri..."}))
File "/google/src/cloud/user/select/google3/myproject/defs.bzl", line 4, in
  my_custom_blaze_macro
my_config_string.upper()
type 'select' has no method upper().
```

Booleans are a special case that fail silently, so you should be particularly
vigilant with them:

```
$ cat myproject/defs.bzl:
def my_boolean_macro(boolval):
  print("TRUE" if boolval else "FALSE")

$ cat myproject/BUILD:
load("//myproject:defx.bzl", "my_boolean_macro")
my_boolean_macro(
    boolval = select({
        "//tools/target_cpu:k8": True,
        "//tools/target_cpu:ppc": False,
    }),
)

$ blaze build //myproject:all --cpu=k8
DEBUG: /google/src/cloud/user/select/google3/myproject/defs.bzl:4:3: TRUE.
$ blaze build //myproject:all --cpu=ppc
DEBUG: /google/src/cloud/user/select/google3/myproject/defs.bzl:4:3: TRUE.
```

This happens because Skylark macros don't understand the contents of select(),
so what they're really evaluting is the select() object itself. According to
[Pythonic](https://docs.python.org/release/2.5.2/lib/truth.html) and [Skylark]
(https://b.corp.google.com/issues/28019197#comment12) design standards, all
objects aside from a very small number of exceptions automatically return true.


### Can I read select() like a dict in Skylark?
go/inspectable-select

Fine. Skylark macros [can't](#skylark-macros-select) evaluate select(s) because
macros are evaluated before Blaze knows what the command line flags are.

Can macros at least read the select()'s dictionary, say, to add an extra suffix
to each branch?

Conceptually this is possible. But this isn't yet implemented and is not being
prioritized. Ping b/34460584 if you'd like to request prioritization (state
the impact as strongly as you can to help us calibrate). But keep in mind the
Blaze team has to weigh this against other urgent priorities.

Offers to help would be given enthusiastic attention.

What you *can* do today is prepare a straight dictionary, then feed it into a
select():

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
        "//tools/target_cpu:k8": "k8 mode",
    },
)

$ blaze build //testapp:selecty --cpu=k8 && cat blaze-genfiles/testapp/selecty.out
k8 mode WITH SUFFIX
```

If you'd like to support both select() and native types, you can do this:

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
      cmd = "echo " + cmd_suffix + "> $@")
```

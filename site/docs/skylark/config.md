---
layout: documentation
title: Starlark Build Configurations
---

# Starlark Build Configurations

User-definable configuration is coming to Starlark!

This makes it possible to:

* define custom flags for your project, obsoleting the need for
[`--define`](configurable-attributes.html#custom-keys)
* write
[transitions](skylark/lib/transition.html#modules.transition)
  to configure deps in different configurations     than their parents (e.g.`--     compilation_mode=opt` or `--cpu=arm`)
* bake better defaults into rules (e.g. automatically build `//my:android_app`
  with a specified SDK)

and more, all completely from .bzl files
(no Bazel release required).
<!-- [TOC] -->

## Current Status

As of Q2'19, this effort is
[partially rolled out](https://github.com/bazelbuild/bazel/issues/5574#issuecomment-458349702)
. Much functionality is guarded while we work out concerns about
memory and performance
at scale.

Related issues:
* [#5574](https://github.com/bazelbuild/bazel/issues/5574) - Starlark support for custom configuration transitions
* [#5575](https://github.com/bazelbuild/bazel/issues/5575) - Starlark support for multi-arch ("fat") binaries
* [#5577](https://github.com/bazelbuild/bazel/issues/5577) - Starlark support for custom build flags
* [#5578](https://github.com/bazelbuild/bazel/issues/5578) - Configuration doesn't block native ->
Skylark rules migration

## User-defined Build Settings
A build setting is a single piece of
[configuration](skylark/rules.html#configurations)
information. Think of a configuration as a key/value map. Setting `--cpu=ppc`
and `--copt="-DFoo"` produces a configuration that looks like
`{cpu: ppc, copt: "-DFoo"}`. Each entry is a build setting.

Traditional flags like `cpu` and `copt` are native settings i.e.
their keys are defined and their values are set inside native bazel java code.
Bazel users can only read and write them via the command line
and other APIs maintained natively. Changing native flags, and the APIs
that expose them, requires a bazel release. User-defined build
settings are defined in `.bzl` files (and thus, don't need a bazel release to
register changes). They also can be set via the command line
(if they're designated as `flags`, see more below), but can also be
set via [user-defined transitions](#user-defined-transitions).

### Defining Build Settings

#### The `build_setting` `rule()` Parameter
Build settings are rules like any other rule and are differentiated using the
Starlark `rule()` function's `build_setting` [attribute](skylark/lib/globals.html#rule.build_setting).

```python
# example/buildsettings/build_settings.bzl
string_flag = rule(
    implementation = _impl,
    build_setting = config.string(flag = True)
)
```

The `build_setting` attribute takes a function that designates the type of the
build setting. The type is limited to a set of basic Starlark types like
`bool` and `string`. See the `config` module [documentation](skylark/lib/config.html)
for details. More complicated typing can be done in the rule's implementation
function. More on this below.

The `config` function also takes an optional boolean parameter, `flag`, which is
set to false by default. if `flag` is set to true, the build setting can be set
on the command line by users as well as internally by rule writers via default
values and
[transitions](skylark/lib/transition.html#modules.transition)
.  Not all settings should be settable by
users. For example if you as a rule writer have some debug mode that you'd like
to turn on inside test rules, you don't want to give users the ability to
indiscriminately turn on that feature inside other non-test rules.

#### Using `ctx.build_setting_value`
Like all rules, build setting rules have [implementation functions](skylark/rules.html#implementation-function).
The basic Starlark-type value of the build settings can be accessed via the
`ctx.build_setting_value` method. This method is only available to [`ctx`](skylark/lib/ctx.html)
objects of build setting rules. These implementation methods can directly
forward the build settings value or do additional work on it, like type checking
or more complex struct creation. Here's how you would implement an `enum`-typed
build setting:

```python
# example/buildsettings/build_settings.bzl
TemperatureProvider = provider(fields = ['type'])

temperatures = ["HOT", "LUKEWARM", "ICED"]

def _impl(ctx):
    raw_temperature = ctx.build_setting_value
    if raw_temperature not in temperatures:
        fail(ctx.label + " build setting allowed to take values "
             + temperatures + " but was set to unallowed value "
             + raw_temperature)
    return TemperatureProvider(type = value)

temperature = rule(
    implementation = _impl,
    build_setting = config.string(flag = true)
)
```

Note: if a rule depends on a build setting, it will receive whatever providers
the build setting implementation function returns, like any other dependency.
But all other references to the value of the build setting (e.g. in transitions)
will see its basic Starlark-typed value, not this post implementation function
value.

#### Instantiating Build Settings
Rules defined with the `build_setting` parameter have an implicit mandatory
`build_setting_default` attribute. This attribute takes on the same type as
declared by the `build_setting` param.

```python
# example/buildsettings/build_settings.bzl
FlavorProvider = provider(fields = ['type'])

def _impl(ctx):
    return FlavorProvider(type = ctx.build_setting_value)

flavor = rule(
    implementation = _impl,
    build_setting = config.string(flag = true)
)
```

```python
# example/buildsettings/BUILD
load("//example/buildsettings:build_settings.bzl", "flavor")
flavor(
    name = "favorite_flavor",
    build_setting_default = "APPLE"
)
```

TODO(bazel-team): Implement common build settings rules and providers for simple
cases where the implementation just fowards the value.

### Using Build Settings

#### Depending on Build Settings
If a target would like to read a piece of configuration information, it can
directly depend on the build setting via a regular attribute dependency.

```python
# example/rules.bzl
load("//example/buildsettings:build_settings.bzl", "FlavorProvider")
def _rule_impl(ctx):
    if ctx.attrs.flavor[FlavorProvider].type == "ORANGE":
        ...

drink_rule = rule(
    implementation = _rule_impl,
    attrs = {
        "flavor": attr.label()
    }
)
```

```python
# example/BUILD
load("//example:rules.bzl", "drink_rule")
load("//example/buildsettings:build_settings.bzl", "flavor")
flavor(
    name = "favorite_flavor",
    build_setting_default = "APPLE"
)
drink_rule(
    name = "my_drink",
    flavor = ":favorite_flavor",
)
```

Languages may wish to create a canonical set of build settings which all rules
for that language depend on. Though the native concept of `fragments` no longer
exists as a hardcoded object in Starlark configuration world, one way to
translate this concept would be to use sets of common implicit attributes. For
example:

```python
# kotlin/rules.bzl
_KOTLIN_CONFIG = {
    "_compiler": attr.label(default = "//kotlin/config:compiler-flag"),
    "_mode": attr.label(default = "//kotlin/config:mode-flag"),
    ...
}

...

kotlin_library = rule(
    implementation = _rule_impl,
    attrs = dicts.add({
        "library-attr": attr.string()
    }, _KOTLIN_CONFIG)
)

kotlin_binary = rule(
    implementation = _binary_impl,
    attrs = dicts.add({
        "binary-attr": attr.label()
    }, _KOTLIN_CONFIG)

```

#### Settings Build Settings on the command line
Build settings are set on the command line like any other flag. Boolean build
settings understand no-prefixes and both equals and space syntaxes are supported.
The name of build settings is their full target path:

```shell
$ bazel build //my/target --//example:favorite_flavor="PAMPLEMOUSSE"
```

There are plans to implement shorthand mapping of flag labels so users don't
need to use their entire target path each time i.e.:

```shell
$ bazel build //my/target --cpu=k8 --noboolean_flag
```

instead of

```shell
$ bazel build //my/target --//third_party/bazel/src/main:cpu=k8 --no//my/project:boolean_flag
```

### Label-typed Build Settings

Unlike other build settings, label-typed settings cannot be defined using the
`build_setting` rule parameter. Instead, bazel has two built-in rules:
`label_flag` and `label_setting`. These rules forward the providers of the
actual target to which the build setting is set. `label_flag` and
`label_setting` can be read/written by transitions and `label_flag` can be set
by the user like other `build_setting` rules can. Their only difference is they
can't customely defined.

Label-typed settings will eventually replace the functionality of late-bound
defaults. Late-bound default attributes are Label-typed attributes whose
final values can be affected by configuration. In Starlark, this will replace
the [configuration_field](skylark/lib/globals.html#configuration_field) API.

```python
# example/rules.bzl
MyProvider = provider(field = ["my_field"])

def _dep_impl(ctx):
    return MyProvider(my_field = "yeehaw")

dep_rule = rule(
    implementation = _dep_impl
)

def _parent_impl(ctx):
    if ctx.attr.my_field_provider[MyProvider] == "cowabunga":
        ...

parent_rule = rule(
    implementation = _parent_impl,
    attrs = { "my_field_provider": attr.label() }
)

```

```python
# example/BUILD
load("//example:rules.bzl", "dep_rule", "parent_rule")

dep_rule(name = "dep")

parent_rule(name = "parent", my_field_provider = ":my_field_provider")

label_flag(
    name = "my_field_provider",
    build_setting_default = ":dep"
)
```

TODO(bazel-team): Expand supported build setting types.

## Build Settings and Select
Users can configure attributes on build settings by using
[`select()`](be/functions.html#select). Build setting targets can be passed to the
`flag_values` attribute of `config_setting`. The value to match to the
configuration is passed as a `String` then parsed to the type of the build
setting for matching.

```python
config_setting(
    name = "my_config",
    flag_values = {
        "//example:favorite_flavor": "MANGO"
    }
)
```


## User-defined Transitions
A configuration
[transition](skylark/lib/transition.html#modules.transition)
is how we change configuration of
configured targets in the build graph.

### Defining Transitions in Starlark
Transitions define configuration changes between rules. For example, a request
like “compile my dependency for a different CPU than its parent” is handled by a
transition.

Formally, a transition is a function from an input configuration to one or more
output configurations. Most transitions are 1:1 e.g. "override the input
configuration with `--cpu=ppc`". 1:2+ transitions can also exist but come
with special restrictions.

In Starlark, transitions are defined much like rules, with a defining
`transition()` [function](skylark/lib/transition.html) and an implementation function.

```python
# example/transitions/transitions.bzl
def _impl(settings, attr):
    _ignore = (settings, attr)
    return {"//example:favorite_flavor" : "MINT"}

hot_chocolate_transition = transition(
    implementation = _impl,
    inputs = [],
    outputs = ["//example:favorite_flavor"]
)
```
The `transition()` function takes in an implementation function, a set of
build settings to read(`inputs`), and a set of build settings to write
(`outputs`). The implementation function has two parameters, `settings` and
`attr`. `settings` is a dictionary {`String`:`Object`} of all settings declared
in the `inputs` parameter to `transition()`.

`attr` is a dictionary of attributes and values of the rule to which the
transition is attached. When attached as an
[outgoing edge transition](#outgoing-edge-transitions), the values of these
attributes are all configured i.e. post-select() resolution. When attached as
an [incoming edge transition](#incoming-edge-transitions), `attr` does not
include any attributes that use a selector to resolve their value. If an
incoming edge transition on `--foo` reads attribute `bar` and then also
selects on `--foo` to set attribute `bar`, then there's a chance for the
incoming edge transition to read the wrong value of `bar` in the transition.

Note: since transitions are attached to rule definitions and `select()`s are
attached to rule instantiations (i.e. targets), errors related to `select()`s on
read attributes will pop up when users create targets rather than when rules are
written. It may be worth taking extra care to communicate to rule users which
attributes they should be wary of selecting on or taking other precautions.

The implementation function must return a dictionary (or list of
dictionaries, in the case of
transitions with multiple output configurations)
of new build settings values to apply. The returned dictionary keyset(s) must
contain exactly the set of build settings passed to the `outputs`
parameter of the transition function. This is true even if a build setting is
not actually changed over the course of the transition - its original value must
be explicitly passed through in the returned dictionary.

#### Defining 1:2+ Transitions
[Outgoing edge transition](#outgoing-edge-transitions) can map a single input
configuration to two or more output configurations. These are defined in
Starlark by returning a list of dictionaries in the transition implementation
function.

```python
# example/transitions/transitions.bzl
def _impl(settings, attr):
    _ignore = (settings, attr)
    return [
        {"//example:favorite_flavor" : "LATTE"},
        {"//example:favorite_flavor" : "MOCHA"},
    ]

coffee_transition = transition(
    implementation = _impl,
    inputs = [],
    outputs = ["//example:favorite_flavor"]
)
```

### Attaching Transitions
Transitions can be attached in two places: incoming edges and outgoing edges.
Effectively this means rules can transition their own configuration (incoming
edge transition) and transition their dependencies' configurations (outgoing
edge transition).

#### Incoming Edge Transitions
Incoming edge transitions are activated by attaching a `transition` object
(created by `transition()`) to `rule()`'s `cfg` parameter:

```python
# example/rules.bzl
load("example/transitions:transitions.bzl", "hot_chocolate_transition")
drink_rule = rule(
    implementation = _impl,
    cfg = hot_chocolate_transition,
    ...
```

Incoming edge transitions must be 1:1 transitions.

### Outgoing Edge Transitions
Outgoing edge transitions are activated by attaching a `transition` object
(created by `transition()`) to an attribute's `cfg` parameter:

```python
# example/rules.bzl
load("example/transitions:transitions.bzl", "coffee_transition")
drink_rule = rule(
    implementation = _impl,
    attrs = { "dep": attr.label(cfg = coffee_transition)}
    ...
```
Outgoing edge transitions can be 1:1 or 1:2+.

### Transitions on Native Options
WARNING: This feature will be deprecated soon. Use at your own risk.

Starlark transitions can also declare reads and writes on native options via
a special prefix to the option name.

```python
# example/transitions/transitions.bzl
def _impl(settings, attr):
    _ignore = (settings, attr)
    return {"//command_line_option:cpu": "k8"}

cpu_transition = transition(
    implementation = _impl,
    inputs = [],
    outputs = ["//command_line_option:cpu"]
```

## Integration with Platforms and Toolchains
Many native flags today, like `--cpu` and `--crosstool_top` are related to
toolchain resolution. In the future, explicit transitions on these types of
flags will likely be replaced by transitioning on the
[target platform](platforms.html)

## Also see:

 * [Starlark Build Configuration](https://docs.google.com/document/d/1vc8v-kXjvgZOdQdnxPTaV0rrLxtP2XwnD2tAZlYJOqw/edit?usp=sharing)
 * [Bazel Configurability Roadmap](https://bazel.build/roadmaps/configuration.html)

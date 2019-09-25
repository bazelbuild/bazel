---
layout: documentation
title: Building With Platforms
---

# Building With Platforms

- [Overview](#overview)
- [Background](#background)
- [Goal](#goal)
- [Should I use platforms?](#should-i-use-platforms)
- [API review](#api-review)
- [Status](#status)
  - [Common platform properties](#common-platform-properties)
  - [Default platforms](#default-platforms)
  - [C++](#c)
  - [Java](#java)
  - [Android](#android)
  - [Go](#go)
  - [Other languages](#other-languages)
  - [select()](#select)
  - [Starlark transitions](#starlark-transitions)
- [How to use platforms today](#how-to-use-platforms-today)
- [Questions](#questions)
- [See also](#see-also)

## Overview

Bazel has sophisticated support for modeling [platforms](platforms.html) and
[toolchains](toolchains.html). Integrating this into real projects requires
coherent cooperation between project and library owners, rule maintainers,
and core Bazel devs.

This page summarizes the arguments for using platforms and shows how to
navigate these relationships for maximum value with minimum cognitive
overhead.

**In short:** the core APIs are available but the rule and depot migrations required
to make them work universally are ongoing. This means you *may* be able to use
platforms and toolchains with your project, with some work. But you have to
explicitly opt your project in.

For more formal documentation, see:

* [Platforms](platforms.html)
* [Toolchains](toolchains.html)

## Background

*Platforms* and *toolchains* were introduced to *standardize* the need for
 software projects to target different kinds of computers with different
 language-appropriate tools.

This is a relatively recent addition to Bazel. It was
[inspired](https://blog.bazel.build/2019/02/11/configurable-builds-part-1.html)
by the observation that language maintainers were *already* doing this in ad hoc
and incompatible ways. For example, C++ rules use `--cpu` and `--crosstool_top`
to set a build's target CPU and C++ toolchain. Neither of these correctly models a
"platform". Historic attempts to use them for that inevitably led to awkward and
inaccurate build APIs. They also don't say anything about Java  toolchains,
which evolved their own independent interface with `--java_toolchain`.

Bazel aims to excel at large, mixed-language, multi-platform projects. This
demands more principled support for these concepts, including clear APIs that
bind rather than diverge languages and projects. This is what the new platform
and toolchain APIs achieve.

### Migration

These APIs aren't enough for all projects to use platforms. We also have to
retire the old APIs. This isn't trivial because all of a project's languages,
toolchains, dependencies, and `select()`s have to support the new APIs. This
requires an *ordered migration sequence* to keep projects working correctly.

For example, Bazel's [C++](/versions/master/bazel-and-cpp.html) rules aleady
support platforms while the [Android](/versions/master/bazel-and-android.html)
rules don't. *Your* C++ project may not care about Android. But others may. So
it's not yet safe to globally enable platforms for all C++ builds.

The thrust of this page describes this migration sequence and how and when your
projects can fit in.

## Goal

Bazel's platform migration is complete when all projects can build with the form:

```sh
$ bazel build //:myproject --platforms=//:myplatform
```

This implies:

1. The rules your project uses can infer correct toolchains from
`//:platform`.
1. The rules your project's dependencies use can infer correct toolchains
from `//:myplatform`.
1. *Either* the projects depending on yours support `//:myplatform` *or* your
project supports the legacy APIs (like `--crosstool_top`).
1. `//:myplatform` references
[common declarations](https://github.com/bazelbuild/platforms#motivation) of
`CPU`, `OS`, and other generic concepts that support automatic cross-project
compatibility.
1. All relevant projects'
[`select()`s](https://docs.bazel.build/versions/master/configurable-attributes.html)
understand the machine properties implied by `//:myplatform`.
1. `//:myplatform` is defined in a clear, reusable place: in your project's
repo if the platform is unique to your project, otherwise somewhere all projects
that may use this platform can find. 

As soon as this goal is achieved, we'll remove the old APIs and make this *the*
way projects select platforms and toolchains.

## Should I use platforms?
Yes. Bazel's platform and toolchain APIs are a major upgrade over legacy ways to
select toolchains and perform multi-platform builds. The real question is, given
the migration work required to make platform support ubiquitous, *when* should
your project fit in? 

The answer varies across projects. You should opt yours in when the value added
outweighs current costs:

### Value
* "just works" for end users

### Costs
* rule / toolchain maintainers have to understand this


## API review
A [`platform`](be/platform.html#platform) is a collection of
[`constraint_value`](be/platform.html#constraint_value)s: 

```python
platform(
    name = "myplatform",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:arm",
    ],
)
```

A [`constraint_value`](be/platform.html#constraint_value) is a machine
property. Values of the same "kind" are grouped under a common
[`constraint_setting`](be/platform.html#constraint_setting): 

```python
constraint_setting(name = "os")
constraint_value(
    name = "linux",
    constraint_setting = ":os",
)
constraint_value(
    name = "mac",
    constraint_setting = ":os",
)
```

A [`toolchain`](toolchains.html) is a [Starlark rule](skylark/rules.html). Its
attributes declare a language's tools (like `compiler =
"//mytoolchain:custom_gcc"`). Its [providers](skylark/rules.html#providers) pass
this information to rules that need to build with these tools.

Toolchains declare the `constraint_value`s of machines they can
[target](be/platform.html#toolchain.target_compatible_with)
(`target_compatible_with = ["@platforms//os:linux"]`) and machines their tools can
[run on](be/platform.html#toolchain.exec_compatible_with)
(`exec_compatible_with = ["@platforms//os:mac"]`).

When building `$ bazel build //:myproject --platforms=//:myplatform`, Bazel
automatically selects a toolchain that can run on the build machine and
build binaries for `//:myplatform`. This is known as *toolchain resolution*.

The set of available toolchains can be registered in the `WORKSPACE` with
[`register_toolchains`](skylark/lib/globals.html#register_toolchains) or at the
command line with [`--extra_toolchains`](command-line-reference.html#flag--extra_toolchains).

See [here](toolchains.html) for a deeper dive.

## Status

Current platform support varies among languages. All of Bazel's major rules are
moving to platforms. But this process will take time. This is for three main reasons:

1. Rule logic must be updated to get tool info from the new [toolchain
API](toolchains.html) (`ctx.toolchains`) and stop reading legacy settings like
`--cpu` and `--crosstool_top`. This is relatively straightforward.

1. Toolchain maintainers must define toolchains and make them accessible to
   users (in GitHub repositories and `WORKSPACE` entries). This is technically
   straightforward but must be intelligently organized to maintain an easy user
   experience.

   Platform definitions are also necessary (unless you build for the same machine
   Bazel runs on). But we generally expect projects to define their own platforms.

1. Existing projects must be migrated. `select()`s and
[transitions](skylark/config.html#user-defined-transitions) also have to be
migrated. This is the biggest challenge. It's particularly challenging for
multi-language projects (which will fail if *all* languages can't read
`--platforms`).

If you're designing a new rule set, we *strongly* recommend you support
platforms from the beginning. This automatically makes your rules compatible
with other rules and projects, with increasing value as the platform API becomes
more ubiquitious. 

More details:


### Common platform properties
Platform properties like `OS` and `CPU` that are common across projects should
be declared in a standard, centralized place. This encourages cross-project
and cross-language compatibility.

For example, if *MyApp* has a `select()` on `constraint_value`
`@myapp//cpus:arm` and *SomeCommonLib* has a `select()` on
`@commonlib//constraints:arm`, these trigger their "arm" modes with incompatible
criteria.

Globally common properties are declared in the
[`@platforms`](https://github.com/bazelbuild/platforms) repo (so the canonical
label for the above example is `@platforms//cpu:arm`). Language-common
properties should be declared in the repos of their respective languages.

### Default platforms

Generally, project owners should define explicit
[platforms](platforms.html#defining-constraints-and-platforms) to describe the
kinds of machines they want to build for. These are then triggered with
`--platforms`.

When `--platforms` isn't set, Bazel defaults to a `platform` representing the
local build machine. This is auto-generated at `@local_config_platform//:host`
so there's no need to explicitly define it. It maps the local machine's `OS`
and `CPU` with `constraint_value`s declared in
[`@platforms`](https://github.com/bazelbuild/platforms).


### C++
Bazel's C++ rules use platforms when you set
`--incompatible_enable_cc_toolchain_resolution`
([#7260](https://github.com/bazelbuild/bazel/issues/7260)).

This means you can configure a C++ project with

```sh
$ bazel build //:my_cpp_project --platforms=//:myplatform
```

instead of the legacy

```sh
$ bazel build //:my_cpp_project` --cpu=... --crosstool_top=...  --compiler=...
```


* C++-specific constraints in [rules_cc](https://github.com/bazelbuild/rules_cc).
* Android Fat APKs still change with --cpu
* Same as multiarch Apple binaries
* #7260 includes migration instructions.
* Starlark rules depending on C++ toolchain: `toolchains = ["@rules_cc//cc:toolchain_type"]`
* [find_cc_toolchain](https://github.com/bazelbuild/rules_cc/blob/master/cc/find_cc_toolchain.bzl) ([example](https://github.com/bazelbuild/rules_cc/blob/master/examples/my_c_compile/my_c_compile.bzl))
* [example of iOS dependency](https://github.com/bazelbuild/bazel/issues/8716#issuecomment-507230303)

### Java
* Replace --java_toolchain, --host_java_toolchain, --javabase, --host_javabase.
*  --incompatible_use_toolchain_resolution_for_java_rules
(https://github.com/bazelbuild/bazel/issues/7849)
* blocked on downstream projects failing
* Add relevant toolchain definitions? https://github.com/bazelbuild/rules_java/pull/8

### Android
* @ahumesky and @jin said that for Android rules, toolchains are defined in a
 standard place and it's highly unusual for users to use a custom
 --crosstool_top. So it may be realistic to define platforms and platform
 mappings alongside these toolchains. See here for early precedent. Apple
 rules are presumably similar.
* https://docs.bazel.build/versions/master/android-ndk.html#integration-with-platforms-and-toolchains



### Go
### Other languages
### `select()`
### Starlark transitions

## How to use platforms today
### Platform mappings
* Tracking bug: [#6426](https://github.com/bazelbuild/bazel/issues/6426)
* [example fix for iOS / C++ breakage](https://github.com/bazelbuild/bazel/issues/8716#issuecomment-516572378)

## Questions
## See also



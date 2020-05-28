---
layout: documentation
title: Building with platforms
---

# Building with platforms

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

For example, Bazel's
[C++](/versions/master/bazel-and-cpp.html)
rules aleady support platforms while the
[Android](/versions/master/bazel-and-android.html)
rules don't. *Your* C++ project may not care about Android. But others may. So
it's not yet safe to globally enable platforms for all C++ builds.

The thrust of this page describes this migration sequence and how and when your
projects can fit in.

## Goal
Bazel's platform migration is complete when all projects build with the form:

```sh
$ bazel build //:myproject --platforms=//:myplatform
```

This implies:

1. The rules your project uses can infer correct toolchains from
`//:myplatform`.
1. The rules your project's dependencies use can infer correct toolchains
from `//:myplatform`.
1. *Either* the projects depending on yours support `//:myplatform` *or* your
project supports the legacy APIs (like `--crosstool_top`).
1. `//:myplatform` references
[common declarations](https://github.com/bazelbuild/platforms#motivation)
of `CPU`, `OS`, and other generic concepts that support automatic cross-project
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
If you just want to build or cross-compile a project, you should follow the
project’s official documentation.

If you’re a project, language, or toolchain maintainer, you'll eventually want
to support the new APIs. Whether you wait until the global migration is complete
or opt in early depends on your specific value / cost needs:

### Value
* You can `select()` or choose toolchains on the exact properties you care
  about instead of hard-coded flags like `--cpu`. For example, multiple CPUs
  can support the [same instruction set](https://en.wikipedia.org/wiki/SSE4).
* More correct builds. If you `select()` with `--cpu` in the above example, then
  add a new CPU that supports the same instruction set, the `select()`
  fails to recognize the new CPU. But a `select()` on platforms remains accurate.
* Simpler user experience. All projects understand:
  `--platforms=//:myplatform`. No need for multiple language-specific
  flags on the command line.
* Simpler language design. All languages share a common API for defining
  toolchains, using toolchains, and selecting the right toolchain for a platform.

### Costs
* Dependent projects that don't yet support platforms might not automatically work
  with yours.
* Making them work may require [additional temporary maintenance](#platform-mappings).
* Co-existence of new and legacy APIs requires more careful user guidance to
  avoid confusion.
* Canonical definitions for [common properties](#common-platorm-properties) like
  `OS` and `CPU` are still evolving and may require extra initial contributions.
* Canonical definitions for language-specific toolchains are still evolving and
  may require extra initial contributions.


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
   users (in GitHub repositories and `WORKSPACE` entries).
   This is technically straightforward but must be intelligently organized to
   maintain an easy user experience.

   Platform definitions are also necessary (unless you build for the same machine
   Bazel runs on). But we generally expect projects to define their own platforms.

1. Existing projects must be migrated. `select()`s and
[transitions](skylark/config.html#user-defined-transitions) also have to be
migrated. This is the biggest challenge. It's particularly challenging for
multi-language projects (which may fail if *all* languages can't read
`--platforms`).

If you're designing a new rule set, we *strongly* recommend you support
platforms from the beginning. This automatically makes your rules compatible
with other rules and projects, with increasing value as the platform API becomes
more ubiquitious.

Details:

### Common platform properties
Platform properties like `OS` and `CPU` that are common across projects should
be declared in a standard, centralized place. This encourages cross-project
and cross-language compatibility.

For example, if *MyApp* has a `select()` on `constraint_value`
`@myapp//cpus:arm` and *SomeCommonLib* has a `select()` on
`@commonlib//constraints:arm`, these trigger their "arm" modes with incompatible
criteria.

Globally common properties are declared in the
[`@platforms`](https://github.com/bazelbuild/platforms) repo
(so the canonical label for the above example is `//third_party/bazel_platforms//cpu:arm`).
Language-common properties should be declared in the repos of their respective
languages.

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
Bazel's C++ rules use platforms to select toolchains when you set
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

If your project is pure C++ and not depended on by non-C++ projects, you can use
this mode safely as long as your [`select`](#select)s and
[transitions](#transitions) also work with platforms. See
[#7260](https://github.com/bazelbuild/bazel/issues/7260) and [Configuring C++
toolchains](tutorial/cc-toolchain-config.html) for further migration guidance.

This mode is not enabled by default. This is because Android and iOS projects
still configure C++ dependencies with `--cpu` and `--crosstool_top`
([example](https://github.com/bazelbuild/bazel/issues/8716#issuecomment-507230303)). Enabling
it requires adding platform support for Android and iOS.

### Java
Bazel's Java rules use platforms to select toolchains.

This replaces legacy flags `--java_toolchain`, `--host_java_toolchain`,
`--javabase`, and `--host_javabase`.

[PR #8](https://github.com/bazelbuild/rules_java/pull/8) defines the Java-specific
`constraint_value`s, toolchains, and other settings that make migration
practical. This mode will be enabled by default after those changes are
committed.

### Android
Bazel's Android rules do not yet support platforms to select Android toolchains.

They do support setting `--platforms` to select NDK toolchains: see
[here](android-ndk.html#integration-with-platforms-and-toolchains).

Most importantly,
[`--fat_apk_cpu`](android-ndk.html#integration-with-platforms-and-toolchains),
which builds multi-architecture fat APKs, does not work with platform-enabled
C++. This is because it sets legacy flags like `--cpu` and `--crosstool_top`,
which platform-enabled C++ rules don't read. Until this is migrated, using
`--fat_apk_cpu` with `--platforms` requires [platform
mappings](#platform-mappings).

### Apple
Bazel's Apple rules do not yet support platforms to select Apple toolchains.

They also don't support platform-enabled C++ dependencies because they use the
legacy `--crosstool_top` to set the C++ toolchain. Until this is migrated, you
can mix Apple projects with platorm-enabled C++ with [platform
mappings](#platform-mappings)
([example](https://github.com/bazelbuild/bazel/issues/8716#issuecomment-516572378)).

### Other languages
* Bazel's [Rust rules](https://github.com/bazelbuild/rules_rust) fully support
platforms.
* Bazel's [Go rules](https://github.com/bazelbuild/rules_go) fully support
platforms
([details](https://github.com/bazelbuild/rules_go#how-do-i-cross-compile)).

If you're designing rules for a new language, we *strongly* encourage you to use
platforms to select your language's toolchains. See
the [toolchains documentation](toolchains.html) for a good walkthrough.

[bazel-configurability@google.com](https://groups.google.com/a/google.com/g/bazel-configurability).
or the owners of the appropriate rules.

For discussions on the design and evolution of the platform/toolchain APIs,
contact
[bazel-configurability@google.com](https://groups.google.com/a/google.com/g/bazel-configurability).

## See also

* [Configurable Builds - Part 1](https://blog.bazel.build/2019/02/11/configurable-builds-part-1.html)
* [Platforms](platforms.html)
* [Toolchains](toolchains.html)
* [Bazel Platforms Cookbook](https://docs.google.com/document/d/1UZaVcL08wePB41ATZHcxQV4Pu1YfA1RvvWm8FbZHuW8/)
* [`hlopko/bazel_platforms_examples`](https://github.com/hlopko/bazel_platforms_examples)
* [Example C++ custom toolchain](https://github.com/gregestren/snippets/tree/master/custom_cc_toolchain_with_platforms)

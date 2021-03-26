---
layout: documentation
title: Building with platforms
category: getting-started
---

# Building with Platforms


Bazel has sophisticated support for modeling [platforms][Platforms] and
[toolchains][Toolchains]. Integrating this into real projects requires
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

* [Platforms][Platforms]
* [Toolchains][Toolchains]

## Background
*Platforms* and *toolchains* were introduced to *standardize* the need for
 software projects to target different kinds of computers with different
 language-appropriate tools.

This is a relatively recent addition to Bazel. It was
[inspired][Inspiration]
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
These APIs aren't enough for all projects to use platforms, and the old APIs
have to be retired. This isn't trivial because all of a project's languages,
toolchains, dependencies, and `select()`s have to support the new APIs. This
requires an *ordered migration sequence* to keep projects working correctly.

For example, Bazel's
[C++ Rules] aleady support platforms while the
[Android Rules] don't. *Your* C++ project may not care about Android. But others may. So
it's not yet safe to globally enable platforms for all C++ builds.

The remainder of this page describes this migration sequence and how and when
your projects can fit in.

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
[common declarations][Common Platform Declaration]
of `CPU`, `OS`, and other generic concepts that support automatic cross-project
compatibility.
1. All relevant projects'
[`select()`s][select()]
understand the machine properties implied by `//:myplatform`.
1. `//:myplatform` is defined in a clear, reusable place: in your project's
repo if the platform is unique to your project, otherwise somewhere all projects
that may use this platform can find.

The old APIs will be removed as soon as this goal is achieved and this will
become the standard way projects select platforms and toolchains.

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
* Targets can be [skipped](platforms.html#skipping-incompatible-targets) in the
  build and test phase if they are incompatible with the target platform.

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
A [`platform`][platform Rule] is a collection of
[`constraint_value` targets][constraint_value Rule]:

```python
platform(
    name = "myplatform",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:arm",
    ],
)
```

A [`constraint_value`][constraint_value Rule] is a machine
property. Values of the same "kind" are grouped under a common
[`constraint_setting`][constraint_setting Rule]:

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

A [`toolchain`][Toolchains] is a [Starlark rule]. Its
attributes declare a language's tools (like `compiler =
"//mytoolchain:custom_gcc"`). Its [providers][Starlark Provider] pass
this information to rules that need to build with these tools.

Toolchains declare the `constraint_value`s of machines they can
[target][target_compatible_with Attribute]
(`target_compatible_with = ["@platforms//os:linux"]`) and machines their tools can
[run on][exec_compatible_with Attribute]
(`exec_compatible_with = ["@platforms//os:mac"]`).

When building `$ bazel build //:myproject --platforms=//:myplatform`, Bazel
automatically selects a toolchain that can run on the build machine and
build binaries for `//:myplatform`. This is known as *toolchain resolution*.

The set of available toolchains can be registered in the `WORKSPACE` with
[`register_toolchains`][register_toolchains Function] or at the
command line with [`--extra_toolchains`][extra_toolchains Flag].

See [here][Toolchains] for a deeper dive.

## Status
Current platform support varies among languages. All of Bazel's major rules are
moving to platforms. But this process will take time. This is for three main reasons:

1. Rule logic must be updated to get tool info from the new [toolchain
API][Toolchains] (`ctx.toolchains`) and stop reading legacy settings like
`--cpu` and `--crosstool_top`. This is relatively straightforward.

1. Toolchain maintainers must define toolchains and make them accessible to
   users (in GitHub repositories and `WORKSPACE` entries).
   This is technically straightforward but must be intelligently organized to
   maintain an easy user experience.

   Platform definitions are also necessary (unless you build for the same machine
   Bazel runs on). Generally, projects should define their own platforms.

1. Existing projects must be migrated. `select()`s and
[transitions][Starlark transitions] also have to be
migrated. This is the biggest challenge. It's particularly challenging for
multi-language projects (which may fail if *all* languages can't read
`--platforms`).

If you're designing a new rule set, you must support platforms from the
beginning. This automatically makes your rules compatible with other
rules and projects, with increasing value as the platform API becomes
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
(so the canonical label for the above example is `@platforms//cpu:arm`).
Language-common properties should be declared in the repos of their respective
languages.

### Default platforms
Generally, project owners should define explicit
[platforms][Defining Constraints and Platforms] to describe the
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
toolchains] for further migration guidance.

This mode is not enabled by default. This is because Android and iOS projects
still configure C++ dependencies with `--cpu` and `--crosstool_top`
([example](https://github.com/bazelbuild/bazel/issues/8716#issuecomment-507230303)). Enabling
it requires adding platform support for Android and iOS.

### Java

Bazel's Java rules use platforms and configuration flags to select toolchains.

This replaces legacy flags `--java_toolchain`, `--host_java_toolchain`,
`--javabase`, and `--host_javabase`.

To learn how to use the configuration flags, see the [Bazel and Java](bazel-and-java.md) manual.
For additional information, see the [Design document](https://docs.google.com/document/d/1MVbBxbKVKRJJY7DnkptHpvz7ROhyAYy4a-TZ-n7Q0r4).

If you are still using legacy flags, follow the migration process in [Issue #7849](https://github.com/bazelbuild/bazel/issues/7849).
-->

### Android
Bazel's Android rules do not yet support platforms to select Android toolchains.

They do support setting `--platforms` to select NDK toolchains: see
[here][Android Rules Platforms].

Most importantly,
[`--fat_apk_cpu`][Android Rules Platforms],
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

If you're designing rules for a new language, use platforms
to select your language's toolchains. See the
[toolchains documentation](toolchains.md) for a good walkthrough.

### `select()`
Projects can [`select()`][select()] on
[`constraint_value` targets][constraint_value Rule] but not complete
platforms. This is intentional so that `select()`s supports as wide a variety
of machines as possible. A library with `ARM`-specific sources should support
*all* `ARM`-powered machines unless there's reason to be more specific.

To select on one or more `constraint_value`s, use

```python
config_setting(
    name = "is_arm",
    constraint_values = [
        "@platforms//cpu:arm",
    ],
)
```

This is equivalent to traditionally selecting on `--cpu`:

```python
config_setting(
    name = "is_arm",
    values = {
        "cpu": "arm",
    },
)
```

More details [here][select() Platforms].

`select`s on `--cpu`, `--crosstool_top`, etc. don't understand `--platforms`. When
migrating your project to platforms, you must either convert them to
`constraint_values` or use [platform mappings](#platform-mappings) to support
both styles through the migration window.

### Transitions
[Starlark transitions][Starlark transitions] change
flags down parts of your build graph. If your project uses a transition that
sets `--cpu`, `--crossstool_top`, or other legacy flags, rules that read
`--platforms` won't see these changes.

When migrating your project to platforms, you must either convert changes like
`return { "//command_line_option:cpu": "arm" }` to `return {
"//command_line_options:platforms": "//:my_arm_platform" }` or use [platform
mappings](#platform-mappings) to support both styles through the migration
window.

## How to use platforms today
If you just want to build or cross-compile a project, you should follow the
project's official documentation. It's up to language and project maintainers to
determine how and when to integrate with platforms, and what value that offers.

If you're a project, language, or toolchain maintainer and your build doesn't
use platforms by default, you have three options (besides waiting for the global
migration):

1. Flip on the "use platforms" flag for your project's languages ([if they have
   one](#status)) and do whatever testing you need to see if the projects you care
   about work.

1. If the projects you care about still depend on legacy flags like `--cpu` and
   `--crosstool_top`, use these together with `--platforms`:


   `$ bazel build //:my_mixed_project --platforms==//:myplatform
   --cpu=... --crosstool_top=...`

    This has some maintenance cost (you have to manually make sure the settings
    match). But this should work in the absense of renegade
    [transitions](#transitions).

1. Write [platform mappings](#platform-mappings) to support both styles by
   mapping `--cpu`-style settings to corresponding platforms and vice versa.

### Platform mappings
*Platform mappings* is a temporary API that lets platform-powered and
legacy-powered logic co-exist in the same build through the latter's deprecation
window.

A platform mapping is a map of either a `platform()` to a
corresponding set of legacy flags or the reverse. For example:

```python
platforms:
  # Maps "--platforms=//platforms:ios" to "--cpu=ios_x86_64 --apple_platform_type=ios".
  //platforms:ios
    --cpu=ios_x86_64
    --apple_platform_type=ios

flags:
  # Maps "--cpu=ios_x86_64 --apple_platform_type=ios" to "--platforms=//platforms:ios".
  --cpu=ios_x86_64
  --apple_platform_type=ios
    //platforms:ios

  # Maps "--cpu=darwin --apple_platform_type=macos" to "//platform:macos".
  --cpu=darwin
  --apple_platform_type=macos
    //platforms:macos
```

Bazel uses this to guarantee all settings, both platform-based and
legacy, are consistently applied throughout the build, including through
[transitions](#transitions).

By default Bazel reads mappings from the `platform_mappings` file in your
workspace root. You can also set
`--platform_mappings=//:my_custom_mapping`.

See
[here](https://docs.google.com/document/d/1Vg_tPgiZbSrvXcJ403vZVAGlsWhH9BUDrAxMOYnO0Ls/edit)
for complete details.

## Questions
For general support and questions about the migration timeline, contact
[bazel-discuss@googlegroups.com](https://groups.google.com/forum/#!forum/bazel-discuss)
or the owners of the appropriate rules.

For discussions on the design and evolution of the platform/toolchain APIs,
contact
[bazel-dev@googlegroups.com](https://groups.google.com/forum/#!forum/bazel-dev).

## See also

* [Configurable Builds - Part 1](https://blog.bazel.build/2019/02/11/configurable-builds-part-1.html)
* [Platforms]
* [Toolchains]
* [Bazel Platforms Cookbook](https://docs.google.com/document/d/1UZaVcL08wePB41ATZHcxQV4Pu1YfA1RvvWm8FbZHuW8/)
* [`hlopko/bazel_platforms_examples`](https://github.com/hlopko/bazel_platforms_examples)
* [Example C++ custom toolchain](https://github.com/gregestren/snippets/tree/master/custom_cc_toolchain_with_platforms)

[Platforms]: platforms.md
[Toolchains]: toolchains.md
[Inspiration]: https://blog.bazel.build/2019/02/11/configurable-builds-part-1.html
[C++ Rules]: /versions/master/bazel-and-cpp.html
[Android Rules]: /versions/master/bazel-and-android.html
[Common Platform Declarations]: https://github.com/bazelbuild/platforms#motivation
[select()]: https://docs.bazel.build/versions/master/configurable-attributes.html
[select() Platforms]: configurable-attributes.md#platforms
[platform Rule]: be/platform.html#platform
[constraint_value Rule]: be/platform.html#constraint_value
[constraint_setting Rule]: be/platform.html#constraint_setting
[Starlark rule]: skylark/rules.html
[Starlark provider]: skylark/rules.html#providers
[target_compatible_with Attribute]: be/platform.html#toolchain.target_compatible_with
[exec_compatible_with Attribute]: be/platform.html#toolchain.exec_compatible_with
[register_toolchains Function]: skylark/lib/globals.html#register_toolchains
[extra_toolchains Flag]: command-line-reference.html#flag--extra_toolchains
[Starlark transitions]: skylark/config.html#user-defined-transitions
[Defining Constraints and Platforms]: platforms.md#defining-constraints-and-platforms
[Configuring C++ toolchains]: tutorial/cc-toolchain-config.html
[Android Rules Platforms]: android-ndk.html#integration-with-platforms-and-toolchains

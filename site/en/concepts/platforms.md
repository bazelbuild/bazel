Project: /_project.yaml
Book: /_book.yaml

# Migrating to Platforms

{% include "_buttons.html" %}

Bazel has sophisticated [support](#background) for modeling
[platforms][Platforms] and [toolchains][Toolchains] for multi-architecture and
cross-compiled builds.

This page summarizes the state of this support.

Key Point: Bazel's platform and toolchain APIs are available today. Not all
languages support them. Use these APIs with your project if you can. Bazel is
migrating all major languages so eventually all builds will be platform-based.

See also:

* [Platforms][Platforms]
* [Toolchains][Toolchains]
* [Background][Background]

## Status {:#status}

### C++ {:#cxx}

C++ rules use platforms to select toolchains when
`--incompatible_enable_cc_toolchain_resolution` is set.

This means you can configure a C++ project with:

```posix-terminal
bazel build //:my_cpp_project --platforms=//:myplatform
```

instead of the legacy:

```posix-terminal
bazel build //:my_cpp_project` --cpu=... --crosstool_top=...  --compiler=...
```

This will be enabled by default in Bazel 7.0 ([#7260](https://github.com/bazelbuild/bazel/issues/7260){: .external}).

To test your C++ project with platforms, see
[Migrating Your Project](#migrating-your-project) and
[Configuring C++ toolchains].

### Java {:#java}

Java rules use platforms to select toolchains.

This replaces legacy flags `--java_toolchain`, `--host_java_toolchain`,
`--javabase`, and `--host_javabase`.

See [Java and Bazel](/docs/bazel-and-java) for details.

### Android {:#android}

Android rules use platforms to select toolchains when
`--incompatible_enable_android_toolchain_resolution` is set.

This means you can configure an Android project with:

```posix-terminal
bazel build //:my_android_project --android_platforms=//:my_android_platform
```

instead of with legacy flags like  `--android_crosstool_top`, `--android_cpu`,
and `--fat_apk_cpu`.

This will be enabled by default in Bazel 7.0 ([#16285](https://github.com/bazelbuild/bazel/issues/16285){: .external}).

To test your Android project with platforms, see
[Migrating Your Project](#migrating-your-project).

### Apple {:#apple}

[Apple rules]{: .external} do not support platforms and are not yet scheduled
for support.

You can still use platform APIs with Apple builds (for example, when building
with a mixture of Apple rules and pure C++) with [platform
mappings](#platform-mappings).

### Other languages {:#other-languages}

* [Go rules]{: .external} fully support platforms
* [Rust rules]{: .external} fully support platforms.

If you own a language rule set, see [Migrating your rule set] for adding
support.

## Background {:#background}

*Platforms* and *toolchains* were introduced to standardize how software
projects target different architectures and cross-compile.

This was
[inspired][Inspiration]{: .external}
by the observation that language maintainers were already doing this in ad
hoc, incompatible ways. For example, C++ rules used `--cpu` and
 `--crosstool_top` to declare a target CPU and toolchain. Neither of these
correctly models a "platform". This produced awkward and incorrect builds.

Java, Android, and other languages evolved their own flags for similar purposes,
none of which interoperated with each other. This made cross-language builds
confusing and complicated.

Bazel is intended for large, multi-language, multi-platform projects. This
demands more principled support for these concepts, including a clear
standard API.

### Need for migration {:#migration}

Upgrading to the new API requires two efforts: releasing the API and upgrading
rule logic to use it.

The first is done but the second is ongoing. This consists of ensuring
language-specific platforms and toolchains are defined, language logic reads
toolchains through the new API instead of old flags like `--crosstool_top`, and
`config_setting`s select on the new API instead of old flags.

This work is straightforward but requires a distinct effort for each language,
plus fair warning for project owners to test against upcoming changes.

This is why this is an ongoing migration.

### Goal {:#goal}

This migration is complete when all projects build with the form:

```posix-terminal
bazel build //:myproject --platforms=//:myplatform
```

This implies:

1. Your project's rules choose the right toolchains for `//:myplatform`.
1. Your project's dependencies choose the right toolchains for `//:myplatform`.
1. `//:myplatform` references
[common declarations][Common Platform Declarations]{: .external}
of `CPU`, `OS`, and other generic, language-independent properties
1. All relevant [`select()`s][select()] properly match `//:myplatform`.
1. `//:myplatform` is defined in a clear, accessible place: in your project's
repo if the platform is unique to your project, or some common place all
consuming projects can find it

Old flags like `--cpu`, `--crosstool_top`, and `--fat_apk_cpu` will be
deprecated and removed as soon as it's safe to do so.

Ultimately, this will be the *sole* way to configure architectures.


## Migrating your project {:#migrating-your-project}

If you build with languages that support platforms, your build should already
work with an invocation like:

```posix-terminal
bazel build //:myproject --platforms=//:myplatform
```

See [Status](#status) and your language's documentation for precise details.

If a language requires a flag to enable platform support, you also need to set
that flag. See [Status](#status) for details.

For your project to build, you need to check the following:

1. `//:myplatform` must exist. It's generally the project owner's responsibility
   to define platforms because different projects target different machines.
   See [Default platforms](#default-platforms).

1. The toolchains you want to use must exist. If using stock toolchains, the
   language owners should include instructions for how to register them. If
   writing your own custom toolchains, you need to [register](https://bazel.build/extending/toolchains#registering-building-toolchains) them in your
   `MODULE.bazel` file or with [`--extra_toolchains`](https://bazel.build/reference/command-line-reference#flag--extra_toolchains).

1. `select()`s and [configuration transitions][Starlark transitions] must
  resolve properly. See [select()](#select) and [Transitions](#transitions).

1. If your build mixes languages that do and don't support platforms, you may
   need platform mappings to help the legacy languages work with the new API.
   See [Platform mappings](#platform-mappings) for details.

If you still have problems, [reach out](#questions) for support.

### Default platforms {:#default-platforms}

Project owners should define explicit
[platforms][Defining Constraints and Platforms] to describe the architectures
they want to build for. These are then triggered with `--platforms`.

When `--platforms` isn't set, Bazel defaults to a `platform` representing the
local build machine. This is auto-generated at `@local_config_platform//:host`
so there's no need to explicitly define it. It maps the local machine's `OS`
and `CPU` with `constraint_value`s declared in
[`@platforms`](https://github.com/bazelbuild/platforms){: .external}.

### `select()` {:#select}

Projects can [`select()`][select()] on
[`constraint_value` targets][constraint_value Rule] but not complete
platforms. This is intentional so `select()` supports as wide a variety of
machines as possible. A library with `ARM`-specific sources should support *all*
`ARM`-powered machines unless there's reason to be more specific.

To select on one or more `constraint_value`s, use:

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

`select`s on `--cpu`, `--crosstool_top`, etc. don't understand `--platforms`.
When migrating your project to platforms, you must either convert them to
`constraint_values` or use [platform mappings](#platform-mappings) to support
both styles during migration.

### Transitions {:#transitions}

[Starlark transitions][Starlark transitions] change
flags down parts of your build graph. If your project uses a transition that
sets `--cpu`, `--crossstool_top`, or other legacy flags, rules that read
`--platforms` won't see these changes.

When migrating your project to platforms, you must either convert changes like
`return { "//command_line_option:cpu": "arm" }` to `return {
"//command_line_option:platforms": "//:my_arm_platform" }` or use [platform
mappings](#platform-mappings) to support both styles during migration.
window.

## Migrating your rule set  {:#migrating-your-rule-set}

If you own a rule set and want to support platforms, you need to:

1. Have rule logic resolve toolchains with the toolchain API. See
   [toolchain API][Toolchains] (`ctx.toolchains`).

1. Optional: define an `--incompatible_enable_platforms_for_my_language` flag so
   rule logic alternately resolves toolchains through the new API or old flags
   like `--crosstool_top` during migration testing.

1. Define the relevant properties that make up platform components. See
   [Common platform properties](#common-platform-properties)

1. Define standard toolchains and make them accessible to users through your
   rule's registration instructions ([details](https://bazel.build/extending/toolchains#registering-building-toolchains))

1. Ensure [`select()`s](#select) and
   [configuration transitions](#transitions) support platforms. This is the
   biggest challenge. It's particularly challenging for multi-language projects
   (which may fail if *all* languages can't read `--platforms`).

If you need to mix with rules that don't support platforms, you may need
[platform mappings](#platform-mappings) to bridge the gap.

### Common platform properties {:#common-platform-properties}

Common, cross-language platform properties like `OS` and `CPU` should be
declared in [`@platforms`](https://github.com/bazelbuild/platforms){: .external}.
This encourages sharing, standardization, and cross-language compatibility.

Properties unique to your rules should be declared in your rule's repo. This
lets you maintain clear ownership over the specific concepts your rules are
responsible for.

If your rules use custom-purpose OSes or CPUs, these should be declared in your
rule's repo vs.
[`@platforms`](https://github.com/bazelbuild/platforms){: .external}.

## Platform mappings {:#platform-mappings}

*Platform mappings* is a temporary API that lets platform-aware logic mix with
legacy logic in the same build. This is a blunt tool that's only intended to
smooth incompatibilities with different migration timeframes.

Caution: Only use this if necessary, and expect to eventually  eliminate it.

A platform mapping is a map of either a `platform()` to a
corresponding set of legacy flags or the reverse. For example:

```python
platforms:
  # Maps "--platforms=//platforms:ios" to "--ios_multi_cpus=x86_64 --apple_platform_type=ios".
  //platforms:ios
    --ios_multi_cpus=x86_64
    --apple_platform_type=ios

flags:
  # Maps "--ios_multi_cpus=x86_64 --apple_platform_type=ios" to "--platforms=//platforms:ios".
  --ios_multi_cpus=x86_64
  --apple_platform_type=ios
    //platforms:ios

  # Maps "--cpu=darwin_x86_64 --apple_platform_type=macos" to "//platform:macos".
  --cpu=darwin_x86_64
  --apple_platform_type=macos
    //platforms:macos
```

Bazel uses this to guarantee all settings, both platform-based and
legacy, are consistently applied throughout the build, including through
[transitions](#transitions).

By default Bazel reads mappings from the `platform_mappings` file in your
workspace root. You can also set
`--platform_mappings=//:my_custom_mapping`.

See the [platform mappings design]{: .external} for details.

## API review {:#api-review}

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

A [`toolchain`][Toolchains] is a [Starlark rule][Starlark rule]. Its
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

The set of available toolchains can be registered in the `MODULE.bazel` file
with [`register_toolchains`][register_toolchains Function] or at the
command line with [`--extra_toolchains`][extra_toolchains Flag].

For more information see [here][Toolchains].

## Questions {:#questions}

For general support and questions about the migration timeline, contact
[bazel-discuss]{: .external} or the owners of the appropriate rules.

For discussions on the design and evolution of the platform/toolchain APIs,
contact [bazel-dev]{: .external}.

## See also {:#see-also}

* [Configurable Builds - Part 1]{: .external}
* [Platforms]
* [Toolchains]
* [Bazel Platforms Cookbook]{: .external}
* [Platforms examples]{: .external}
* [Example C++ toolchain]{: .external}

[Android Rules]: /docs/bazel-and-android
[Apple Rules]: https://github.com/bazelbuild/rules_apple
[Background]: #background
[Bazel platforms Cookbook]: https://docs.google.com/document/d/1UZaVcL08wePB41ATZHcxQV4Pu1YfA1RvvWm8FbZHuW8/
[bazel-dev]: https://groups.google.com/forum/#!forum/bazel-dev
[bazel-discuss]: https://groups.google.com/forum/#!forum/bazel-discuss
[Common Platform Declarations]: https://github.com/bazelbuild/platforms
[constraint_setting Rule]: /reference/be/platforms-and-toolchains#constraint_setting
[constraint_value Rule]: /reference/be/platforms-and-toolchains#constraint_value
[Configurable Builds - Part 1]: https://blog.bazel.build/2019/02/11/configurable-builds-part-1.html
[Configuring C++ toolchains]: /tutorials/ccp-toolchain-config
[Defining Constraints and Platforms]: /extending/platforms#constraints-platforms
[Example C++ toolchain]: https://github.com/gregestren/snippets/tree/master/custom_cc_toolchain_with_platforms
[exec_compatible_with Attribute]: /reference/be/platforms-and-toolchains#toolchain.exec_compatible_with
[extra_toolchains Flag]: /reference/command-line-reference#flag--extra_toolchains
[Go Rules]: https://github.com/bazelbuild/rules_go
[Inspiration]: https://blog.bazel.build/2019/02/11/configurable-builds-part-1.html
[Migrating your rule set]: #migrating-your-rule-set
[Platforms]: /extending/platforms
[Platforms examples]: https://github.com/hlopko/bazel_platforms_examples
[platform mappings design]: https://docs.google.com/document/d/1Vg_tPgiZbSrvXcJ403vZVAGlsWhH9BUDrAxMOYnO0Ls/edit
[platform Rule]: /reference/be/platforms-and-toolchains#platform
[register_toolchains Function]: /rules/lib/globals/module#register_toolchains
[Rust rules]: https://github.com/bazelbuild/rules_rust
[select()]: /docs/configurable-attributes
[select() Platforms]: /docs/configurable-attributes#platforms
[Starlark provider]: /extending/rules#providers
[Starlark rule]: /extending/rules
[Starlark transitions]: /extending/config#user-defined-transitions
[target_compatible_with Attribute]: /reference/be/platforms-and-toolchains#toolchain.target_compatible_with
[Toolchains]: /extending/toolchains

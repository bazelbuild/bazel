Project: /_project.yaml
Book: /_book.yaml

# Platforms

{% include "_buttons.html" %}

Bazel can build and test code on a variety of hardware, operating systems, and
system configurations, using many different versions of build tools such as
linkers and compilers. To help manage this complexity, Bazel has a concept of
*constraints* and *platforms*. A constraint is a dimension in which build or
production environments may differ, such as CPU architecture, the presence or
absence of a GPU, or the version of a system-installed compiler. A platform is a
named collection of choices for these constraints, representing the particular
resources that are available in some environment.

Modeling the environment as a platform helps Bazel to automatically select the
appropriate
[toolchains](/extending/toolchains)
for build actions. Platforms can also be used in combination with the
[config_setting](/reference/be/general#config_setting)
rule to write [configurable attributes](/docs/configurable-attributes).

Bazel recognizes three roles that a platform may serve:

*  **Host** - the platform on which Bazel itself runs.
*  **Execution** - a platform on which build tools execute build actions to
   produce intermediate and final outputs.
*  **Target** - a platform on which a final output resides and executes.

Bazel supports the following build scenarios regarding platforms:

*  **Single-platform builds** (default) - host, execution, and target platforms
   are the same. For example, building a Linux executable on Ubuntu running on
   an Intel x64 CPU.

*  **Cross-compilation builds** - host and execution platforms are the same, but
   the target platform is different. For example, building an iOS app on macOS
   running on a MacBook Pro.

*  **Multi-platform builds** - host, execution, and target platforms are all
   different.

Tip: for detailed instructions on migrating your project to platforms, see
[Migrating to Platforms](/concepts/platforms).

## Defining constraints and platforms {:#constraints-platforms}

The space of possible choices for platforms is defined by using the
 [`constraint_setting`](/reference/be/platform#constraint_setting) and
 [`constraint_value`](/reference/be/platform#constraint_value) rules within `BUILD` files. `constraint_setting` creates a new dimension, while
`constraint_value` creates a new value for a given dimension; together they
effectively define an enum and its possible values. For example, the following
snippet of a `BUILD` file introduces a constraint for the system's glibc version
with two possible values.

```python
constraint_setting(name = "glibc_version")

constraint_value(
    name = "glibc_2_25",
    constraint_setting = ":glibc_version",
)

constraint_value(
    name = "glibc_2_26",
    constraint_setting = ":glibc_version",
)
```

Constraints and their values may be defined across different packages in the
workspace. They are referenced by label and subject to the usual visibility
controls. If visibility allows, you can extend an existing constraint setting by
defining your own value for it.

The [`platform`](/reference/be/platforms-and-toolchains#platform) rule introduces a new platform with
certain choices of constraint values. The
following creates a platform named `linux_x86`, and says that it describes any
environment that runs a Linux operating system on an x86_64 architecture with a
glibc version of 2.25. (See below for more on Bazel's built-in constraints.)

```python
platform(
    name = "linux_x86",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
        ":glibc_2_25",
    ],
)
```

Note: It is an error for a platform to specify more than one value of the
same constraint setting, such as `@platforms//cpu:x86_64` and
`@platforms//cpu:arm` for `@platforms//cpu:cpu`.

## Generally useful constraints and platforms {:#useful-constraints-platforms}

To keep the ecosystem consistent, Bazel team maintains a repository with
constraint definitions for the most popular CPU architectures and operating
systems. These are all located in
[https://github.com/bazelbuild/platforms](https://github.com/bazelbuild/platforms){: .external}.

Bazel ships with the following special platform definition:
`@local_config_platform//:host`. This is the autodetected host platform value -
represents autodetected platform for the system Bazel is running on.

## Specifying a platform for a build {:#specifying-build-platform}

You can specify the host and target platforms for a build using the following
command-line flags:

*  `--host_platform` - defaults to `@bazel_tools//platforms:host_platform`
*  `--platforms` - defaults to `@bazel_tools//platforms:target_platform`

## Skipping incompatible targets {:#skipping-incompatible-targets}

When building for a specific target platform it is often desirable to skip
targets that will never work on that platform. For example, your Windows device
driver is likely going to generate lots of compiler errors when building on a
Linux machine with `//...`. Use the
[`target_compatible_with`](/reference/be/common-definitions#common.target_compatible_with)
attribute to tell Bazel what target platform constraints your code has.

The simplest use of this attribute restricts a target to a single platform.
The target will not be built for any platform that doesn't satisfy all of the
constraints. The following example restricts `win_driver_lib.cc` to 64-bit
Windows.

```python
cc_library(
    name = "win_driver_lib",
    srcs = ["win_driver_lib.cc"],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
)
```

`:win_driver_lib` is *only* compatible for building with 64-bit Windows and
incompatible with all else. Incompatibility is transitive. Any targets
that transitively depend on an incompatible target are themselves considered
incompatible.

### When are targets skipped? {:#when-targets-skipped}

Targets are skipped when they are considered incompatible and included in the
build as part of a target pattern expansion. For example, the following two
invocations skip any incompatible targets found in a target pattern expansion.

```console
$ bazel build --platforms=//:myplatform //...
```

```console
$ bazel build --platforms=//:myplatform //:all
```

Incompatible tests in a [`test_suite`](/reference/be/general#test_suite) are
similarly skipped if the `test_suite` is specified on the command line with
[`--expand_test_suites`](/reference/command-line-reference#flag--expand_test_suites).
In other words, `test_suite` targets on the command line behave like `:all` and
`...`. Using `--noexpand_test_suites` prevents expansion and causes
`test_suite` targets with incompatible tests to also be incompatible.

Explicitly specifying an incompatible target on the command line results in an
error message and a failed build.

```console
$ bazel build --platforms=//:myplatform //:target_incompatible_with_myplatform
...
ERROR: Target //:target_incompatible_with_myplatform is incompatible and cannot be built, but was explicitly requested.
...
FAILED: Build did NOT complete successfully
```

Incompatible explicit targets are silently skipped if
`--skip_incompatible_explicit_targets` is enabled.

### More expressive constraints {:#expressive-constraints}

For more flexibility in expressing constraints, use the
`@platforms//:incompatible`
[`constraint_value`](/reference/be/platform#constraint_value) that no platform
satisfies.

Use [`select()`](/reference/be/functions#select) in combination with
`@platforms//:incompatible` to express more complicated restrictions. For
example, use it to implement basic OR logic. The following marks a library
compatible with macOS and Linux, but no other platforms.

Note: An empty constraints list is equivalent to "compatible with everything".

```python
cc_library(
    name = "unixish_lib",
    srcs = ["unixish_lib.cc"],
    target_compatible_with = select({
        "@platforms//os:osx": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
```

The above can be interpreted as follows:

1. When targeting macOS, the target has no constraints.
2. When targeting Linux, the target has no constraints.
3. Otherwise, the target has the `@platforms//:incompatible` constraint. Because
   `@platforms//:incompatible` is not part of any platform, the target is
   deemed incompatible.

To make your constraints more readable, use
[skylib](https://github.com/bazelbuild/bazel-skylib){: .external}'s
[`selects.with_or()`](https://github.com/bazelbuild/bazel-skylib/blob/main/docs/selects_doc.md#selectswith_or){: .external}.

You can express inverse compatibility in a similar way. The following example
describes a library that is compatible with everything _except_ for ARM.

```python
cc_library(
    name = "non_arm_lib",
    srcs = ["non_arm_lib.cc"],
    target_compatible_with = select({
        "@platforms//cpu:arm": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
)
```

### Detecting incompatible targets using `bazel cquery` {:#cquery-incompatible-target-detection}

You can use the
[`IncompatiblePlatformProvider`](/rules/lib/providers/IncompatiblePlatformProvider)
in `bazel cquery`'s [Starlark output
format](/query/cquery#output-format-definition) to distinguish
incompatible targets from compatible ones.

This can be used to filter out incompatible targets. The example below will
only print the labels for targets that are compatible. Incompatible targets are
not printed.

```console
$ cat example.cquery

def format(target):
  if "IncompatiblePlatformProvider" not in providers(target):
    return target.label
  return ""


$ bazel cquery //... --output=starlark --starlark:file=example.cquery
```

### Known Issues

Incompatible targets [ignore visibility
restrictions](https://github.com/bazelbuild/bazel/issues/16044).

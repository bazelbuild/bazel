---
layout: documentation
title: Platforms
---

# Platforms

- [Overview](#overview)
- [Defining constraints and platforms](#defining-constraints-and-platforms)
- [Built-in constraints and platforms](#built-in-constraints-and-platforms)
- [Specifying a platform for a build](#specifying-a-platform-for-a-build)

## Overview

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
[toolchains](toolchains.html)
for build actions. Platforms can also be used in combination with the
[config_setting](be/general.html#config_setting)
rule to write
<a href="configurable-attributes.html"> configurable attributes</a>.

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

## Defining constraints and platforms

The space of possible choices for platforms is defined by using the
 [`constraint_setting`](be/platform.html#constraint_setting) and
 [`constraint_value`](be/platform.html#constraint_value) rules within `BUILD` files. `constraint_setting` creates a new dimension, while
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

The
 [`platform`](be/platform.html#platform) rule introduces a new platform with certain choices of constraint values. The
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

Note that it is an error for a platform to specify more than one value of the
same constraint setting, such as `@platforms//cpu:x86_64` and
`@platforms//cpu:arm` for `@platforms//cpu:cpu`.


## Generally useful constraints and platforms

To keep the ecosystem consistent, Bazel team maintains a repository with
constraint definitions for the most popular CPU architectures and operating
systems. These are all located in
[https://github.com/bazelbuild/platforms](https://github.com/bazelbuild/platforms).

Bazel ships with the following special platform definition:
`@local_config_platform//:host`. This is the autodetected host platform value -
represents autodetected platform for the system Bazel is running on.

## Specifying a platform for a build

You can specify the host and target platforms for a build using the following
command-line flags:

*  `--host_platform` - defaults to `@bazel_tools//platforms:host_platform`

*  `--platforms` - defaults to `@bazel_tools//platforms:target_platform`

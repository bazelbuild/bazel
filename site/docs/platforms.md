---
layout: documentation
title: Platforms
---

# Platforms

- [Overview](#overview)
- [Defining a platform](#defining-a-platform)
- [Built-in constraints and platforms](#built-in-constraints-and-platforms)
- [Specifying a platform for a build](#specifying-a-platform-for-a-build)

## Overview

Bazel can build and test code on a variety of operating systems and hardware
using many different build tools, such as linkers and compilers. These
combinations of software and hardware are what Bazel considers *platforms*.
One major use for specifying a platform for a build is automatic
[toolchain](toolchains.html) selection.

Bazel recognizes the following types of platforms:

*  **Host** - platforms on which Bazel runs.
*  **Execution** - platforms on which build tools execute build actions.
*  **Target** - platforms for which Bazel builds the output.

Bazel supports the following build scenarios regarding platforms:

*  **Single-platform builds** (default) - host, execution, and target platforms
   are the same. For example, building a Linux executable on Ubuntu running on
   an Intel x64 CPU.

*  **Cross-compilation builds** - host and execution platforms are the same, but
   the target platform is different. For example, building an iOS app on macOS
   running on a MacBook Pro.

*  **Multi-platform builds** - host, execution, and target platforms are all
   different.

## Defining a platform

A *Bazel platform* is a named collection of constraints that define a supported
software and/or hardware configuration through name-value pairs. For example, a
constraint can define the CPU architecture, GPU presence, or the specific
version of a build tool, such as a linker or compiler.

You define a platform in a `BUILD` file using the following Bazel rules:

*  [`constraint_setting`](be/platform.html#constraint_setting) - defines a
   constraint.

*  [`constraint_value`](be/platform.html#constraint_value) - defines an allowed
   value for a constraint.

*  [`platform`](be/platform.html#platform) - defines a platform by specifying
   a set of constraints and their values.

The following example defines the `glibc_version` constraint and its two allowed
values. It then defines a platform that uses the `glibc_version` constraint
along with Bazel's [built-in constraints](#built-in-constraints-and-platforms)
for operating systems and CPU architecture:

```python
constraint_setting(name = 'glibc_version')

constraint_value(
    name = 'glibc_2_25',
    constraint_setting = ':glibc_version')

constraint_value(
    name = 'glibc_2_26',
    constraint_setting = ':glibc_version')

platform(
    name = 'linux_x86',
    constraint_values = [
      '@bazel_tools//platforms:linux',
      '@bazel_tools//platforms:x86_64',
      ':glibc_2_25',
    ])
```

Keep the following in mind when defining constraints and platforms that use
them:

*  You can define constraints in any Bazel package within the project.

*  Constraints follow the visibility settings of the package that contains them.

*  You can use constraint values from multiple packages in the same platform
   definition. However, using constraint values that share a constraint setting
   will result in an error.

## Built-in constraints and platforms

Bazel ships with constraint definitions for the most popular CPU architectures
and operating systems.

*  `@bazel_tools//platforms:cpu` defines the following CPU architectures:
   *  `@bazel_tools//platforms:x86_32`
   *  `@bazel_tools//platforms:x86_64`
   *  `@bazel_tools//platforms:ppc`
   *  `@bazel_tools//platforms:arm`
   *  `@bazel_tools//platforms:s390x`
*   `@bazel_tools//platforms:os` defines the following operating systems:
   *  `@bazel_tools//platforms:osx`
   *  `@bazel_tools//platforms:freebsd`
   *  `@bazel_tools//platforms:linux`
   *  `@bazel_tools//platforms:windows`

Bazel also ships with the following platform definitions:

*  `@bazel_tools//platforms:host_platform` - automatically detects the CPU
   architecture and operating system for the host platform.

*  `@bazel_tools//platforms:target_platform` - automatically detects the CPU
   architecture and operating system for the target platform.

In these definitions, the CPU architecture constraint values are pulled from the
`--host_cpu` and `--cpu` flags.

## Specifying a platform for a build

To select a specific host and target platform for a build, use the following
command-line flags:

*  `--host_platform` - defaults to `@bazel_tools//platforms:host_platform`

*  `--platforms` - defaults to `@bazel_tools//platforms:target_platform`

Platforms can also be used with the `config_setting` rule to define configurable
attributes. See [config_setting](be/general.html#config_setting) for more
details.

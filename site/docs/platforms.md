---
layout: documentation
title: Platforms
---

# Platforms

Bazel can build and test code on a variety of platforms, in a few different
configurations. Some of configurations are:

+  Single-platform builds - The default, where Bazel executes on the same
    device as the compiler, and the executable will run on the same device.
+  Cross-compile builds - In this case, Bazel and the compiler run on one
    device, but the final executable is intended for another device. This
    includes cases like building Android or iOS apps, among others.
+  Multi-platform builds - A multi-platform build is where Bazel runs on one
    device, but the compilers and other tools run on other devices.

## Contents

* [Types of Platforms](#types-of-platforms)
* [Platforms](#platforms-1)
  * [Overview](#overview)
  * [Defining Platforms](#defining-platforms)
    * [Constraint Settings and Values](#constraint-settings-and-values)
    * [Platforms](#platforms-2)
  * [Using Platforms](#using-platforms)
    * [Flags](#flags)
    * [Predefined Platforms](#predefined-platforms)
* [Toolchains](#toolchains)
  * [Overview](#overview-1)
  * [Toolchain type](#toolchain-type)
  * [Toolchain rules](#toolchain-rules)
    * [Defining a toolchain rule in Skylark]()
  * [Toolchain definition](#toolchain-definition)
  * [Registering a toolchain](#registering-a-toolchain)
  * [Toolchain Resolution](#toolchain-resolution)
  * [Debugging toolchains](#debugging-toolchains)
  * [Using toolchains in Skylark rules](#using-toolchains-in-skylark-rules)
    * [Toolchains attribute on rule()](#toolchains-attribute-on-rule)
    * [ctx.toolchains](#ctx-toolchains)
* [Select and config\_setting](#select-and-config_setting)


## Types of Platforms

To describe these different scenarios, Bazel has introduced the concept of
"platforms". A platform is a device that can run a program, whether that program
is Bazel itself, a compiler or other tool, or the final built output.

The terminology for these types of platforms are:

* Host Platform - Where Bazel itself is running
* Execution Platform - Where build actions run
* Target Platform - Where the final build outputs will run

Typically, the host and execution platform are the same (except in the case of
remote execution), since build actions are executed locally by Bazel. If you are
building an executable that will run on the same computer where you are using
Bazel, then the target platform is the same as the host platform. When you are
cross-compiling (say, building an Android application), then the target platform
is different from the host platform.

## Platforms

### Overview

Conceptually in Bazel, a platform is a named collection of constraints.
Constraints are just names and values used to define the features of a platform.
Some examples of constraints are the CPU architecture, the operating system, the
presence of a GPU, or the specific version of a compiler or other tool.

### Defining Platforms

Platforms are defined in Bazel BUILD files, using three new rules:

* [constraint_setting](be/platform.html#constraint_setting) - Defines a new constraint
* [constraint_value](be/platform.html#constraint_value) - Adds a possible value to an existing constraint
* [platform](be/platform.html#platform) - Defines a set of constraints and names them as a platform

#### Constraint Settings and Values

Bazel automatically defines some constraints and values for you:

TODO(katre): fix table formatting
<table>
<thead>
<tr>
<th>constraint\_setting</th>
<th>constraint\_value</th>
</tr>
</thead>
<tbody>
<tr>
<td>@bazel\_tools//platforms:cpu</td>
<td>@bazel\_tools//platforms:x86\_32</td>
</tr>
<tr>
<td></td>
<td>@bazel\_tools//platforms:x86\_64</td>
</tr>
<tr>
<td></td>
<td>@bazel\_tools//platforms:ppc</td>
</tr>
<tr>
<td></td>
<td>@bazel\_tools//platforms:arm</td>
</tr>
<tr>
<td></td>
<td>@bazel\_tools//platforms:s390x</td>
</tr>
<tr>
<td>@bazel\_tools//platforms:os</td>
<td>@bazel\_tools//platforms:osx</td>
</tr>
<tr>
<td></td>
<td>@bazel\_tools//platforms:freebsd</td>
</tr>
<tr>
<td></td>
<td>@bazel\_tools//platforms:linux</td>
</tr>
<tr>
<td></td>
<td>@bazel\_tools//platforms:windows</td>
</tr>
</tbody>
</table>

Here are some examples of defining additional constraints:

```bash
constraint_value(
    name = 'aarm64',
    constraint_setting = '@bazel_tools//platforms:cpu')

constraint_value(
    name = 'openbsd',
    constraint_setting = '@bazel_tools//platforms:os')

constraint_setting(name = 'glibc_version')

constraint_value(
    name = 'glibc_2_25',
    constraint_setting = ':glibc_version')

constraint_value(
    name = 'glibc_2_26',
    constraint_setting = ':glibc_version')
```

Constraint values can be defined in any package, although constraint settings
and values are subject to the usual rules on visibility.

#### Platforms

Here are some platforms defined using these constraints:

```bash
platform(
    name = 'linux_x86',
    constraint_values = [
      '@bazel_tools//platforms:linux',
      '@bazel_tools//platforms:x86_64',
      ':glibc_2_25',
    ])

platform(
    name = 'openbsd_arm',
    constraint_values = [
      ':openbsd',
      '@bazel_tools//platforms:arm',
    ])
```

You can mix constraint values from multiple packages in a single platform. Be
careful, however, because you cannot use two constraint values that share a
constraint setting! This will be reported as an error.

### Using Platforms

#### Flags

Being able to define platforms is nice, but what can you do with them?
Currently, the main use of platforms and constraints is to have Bazel
automatically select the right toolchain for your build.

Note: not all sets of rules support toolchains, notably the native C++ and Java
rules do not work with this yet.

The host and target platform can be set via command-line flags:

TODO(katre): fix table formatting
<table>
<thead>
<tr>
<th>--experimental\_host\_platform</th>
<th>Defaults to @bazel\_tools//platforms:host\_platform</th>
</tr>
</thead>
<tbody>
<tr>
<td>--experimental\_platforms</td>
<td>Defaults to @bazel\_tools//platforms:target\_platform</td>
</tr>
</tbody>
</table>

#### Predefined Platforms

There is two special predefined platforms: @bazel\_tools//platforms:host\_platform
and @bazel\_tools//platforms:target\_platform. These platforms automatically
detect the current host and target CPU and OS (based on the "--host\_cpu" and
"--cpu" flags) and sets those two constraint values. All other platforms that
you want to use will need to be added to your BUILD files.

## Toolchains

### Overview

A toolchain is a configuration [provider](skylark/rules.html#providers) that allows
you to tell a rule what compiler, compiler flags, etc to use. The set of
available configuration parameters is up to the rule author.

There are three important definitions for every type of toolchain:

1. The rule-specific toolchain rule, such as go\_toolchain, that defines
    the specific values of the configuration to be used by the rules.
1. The toolchain definition, via the toolchain() rule, which tells Bazel
    which constraints the toolchain meets.
1. The toolchain registration, via the register\_toolchains() WORKSPACE
    function, which tells Bazel that the toolchain is available.

During a build, Bazel will use the current execution and target platforms to
find an appropriate toolchain instance for each rule, based on the toolchain's
declared constraints. This process is called Toolchain Resolution.

### Toolchain type

Every set of rules that uses a toolchain needs to define a unique label, the
toolchain type, that is used by the rules when requesting a valid toolchain. The
toolchain type label is given when the toolchain is defined in the toolchain()
rule, so that Bazel can use it as an input to Toolchain Resolution.

### Toolchain rules

To work with toolchains, a rule author needs to add a rule-specific toolchain
rule, which defines the values of all configuration parameters the rule uses
which change for different execution and target platforms. This rule should
return a [ToolchainInfo provider](skylark/lib/ToolchainInfo.html).

Instances of the toolchain rule will be lazily instantiated by Bazel only when
needed. Because of this, a toolchain rule's dependencies can be as complex as
needed, and even reply on remote repositories, and not affect builds where they
are not used.

#### Defining a toolchain rule in Skylark

To define a new toolchain in Skylark, first you need to determine what
information your rules need. Let's consider the case of a new programming
language: the rules will need the path to the compiler, the path to the system
libraries, and a flag that controls the generated binary's CPU architecture.

Toolchain rules are ordinary Skylark rules that create and return providers.
This example toolchain rule would look like this in Skylark:

```bash
def _my_toolchain_impl(ctx):
  toolchain = platform.ToolchainInfo(
    compiler = ctx.attr.compiler,
    system_lib = ctx.attr.system_lib,
    arch_flag = ctx.attr.arch_flag,
  )
  return [toolchain]

my_toolchain = rule(
    _my_toolchain_impl,
    attrs = {
        'compiler': attr.string(),
        'system_lib': attr.string(),
        'arch_flags': attr.string_list(),
    })
```

And a sample usage would look like:

```bash
my_toolchain(name = 'linux_toolchain_impl',
  compiler = '@remote_linux_repo//compiler:compiler_binary',
  system_lib = '@remote_linux_repo//library:system_library',
  arch_flags = [
    '--arch=Linux',
    '--debug_everything',
  ]
)

my_toolchain(name = 'darwin_toolchain_impl',
  compiler = '@remote_darwin_repo//compiler:compiler_binary',
  system_lib = '@remote_darwin_repo//library:system_library',
  arch_flags = [
    '--arch=Darwin',
    #'--debug_everything', # --debug_everything currently broken on Darwin
  ]
)
```

### Toolchain definition

To accomplish the lazy loading of toolchains, a separate rule called toolchain()
is used. This rule tells Bazel the toolchain type, the execution and target
constraints, and the label of the actual rule-specific toolchain to be used.

Example toolchain definitions:

```bash
toolchain(name = 'linux_toolchain',
    toolchain_type = '//path/to:my_toolchain_type',
    exec_compatible_with = [
        '@bazel_tools//platforms:linux',
        '@bazel_tools//platforms:x86_64'],
    target_compatible_with = [
        '@bazel_tools//platforms:linux',
        '@bazel_tools//platforms:x86_64'],
    toolchain = ':linux_toolchain_impl',
)

toolchain(name = 'darwin_toolchain',
    toolchain_type = '//path/to:my_toolchain_type',
    exec_compatible_with = [
        '@bazel_tools//platforms:darwin',
        '@bazel_tools//platforms:x86_64'],
    target_compatible_with = [
        '@bazel_tools//platforms:darwin',
        '@bazel_tools//platforms:x86_64'],
    toolchain = ':darwin_toolchain_impl',
)
```

### Registering a toolchain

Finally, Bazel needs to know about the toolchain definition, in order to use it.
Registering a toolchain can be done in either the WORKSPACE, or via the
"--experimental\_extra\_toolchains" flag.

Example in the WORKSPACE:

```bash
register_toolchains(
  '//path/to:linux_toolchain',
  '//path/to:darwin_toolchain',
)
```

### Toolchain Resolution

When a target requires a toolchain, Bazel checks the list of registered
toolchains to find the first one that is valid, and creates a dependency from
the target to the specific toolchain that was selected. The process looks like
this:

1. Check every registered toolchain in order. First consider the
    toolchains from the "--experimental\_extra\_toolchains" flag, and then the
    "registered\_toolchains" calls in the WORKSPACE.
1. Does the toolchain match the requested toolchain type? If no, skip this one.
1. Do the execution and target constraints on the toolchain match the
    current execution and target platforms? If not, skip this one.
1. We have a matching toolchain, use it.

Please note that this will always choose the first toolchain that is valid, so
toolchains should be ordered by preference if it is possible that multiple could
match.

### Debugging toolchains

When adding toolchain support to existing rules, it can be difficult to
determine where errors are found and what causes problems with toolchain
resolution. To help aid development, the "--toolchain\_resolution\_debug" flag has
been added. This flag causes the Toolchain Resolution process to be very verbose
about what toolchains are being considered, and why they are being skipped.

### Using toolchains in Skylark rules

#### Toolchains attribute on rule()

In order for rules to use the toolchain, the rule author needs to add the
toolchain type to the rule definition:

```bash
my_library = rule(
    ...
    toolchains = ['//path/to:my_toolchain_type']
    ...)
```

#### ctx.toolchains

When the rule is used, Bazel will check the execution and target platforms, and
choose an appropriate toolchain that matches (or fail with an error message
explaining the lack of toolchains). The rule implementation can then access the
toolchain as:

```bash
def _my_library_impl(ctx):
  toolchain = ctx.toolchains['//path/to:my_toolchain_type']
  command = '%s -l %s %s' % (toolchain.compiler, toolchain.system_lib, toolchain.arch_flag)
  ...
```

## Select and config_setting

Platforms can also be used with the config\_setting rule to define configurable
attributes. See [config_setting](be/general.html#config_setting) for more
details.


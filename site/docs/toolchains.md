---
layout: documentation
title: Toolchains
---

# Toolchains

- [Overview](#overview)
- [Defining a toolchain](#defining-a-toolchain)
   - [Creating a toolchain rule](#creating-a-toolchain-rule)
   - [Creating a toolchain definition](#creating-a-toolchain-definition)
   - [Registering a toolchain](#registering-a-toolchain)
- [Using a toolchain in a new rule](#using-a-toolchain-in-a-rule)
- [Debugging a toolchain](#debugging-a-toolchain)

## Overview

A *Bazel toolchain* is a configuration [provider](skylark/rules.html#providers)
that tells a build rule what build tools, such as compilers and linkers, to use
and how to configure them using parameters defined by the rule's creator.

When a build runs, Bazel performs toolchain resolution based on the specified
[execution and target platforms](platforms.html) to determine and apply the
toolchain most appropriate to that build. It does so by matching the
[constraints](platforms.html#defining-a-platform) specified in the project's
`BUILD` file(s) with the constraints specified in the toolchain definition.

**Note:** Some Bazel rules do not yet support toolchain resolution.

When a target requests a toolchain, Bazel checks the list of registered
toolchains and creates a dependency from the target to the first matching
toolchain it finds. To find a matching toolchain, Bazel does the following:

1.  Looks through the registered toolchains, first from the `--extra_toolchains`
    flag, then from the `register_toolchains` calls in the project's `WORKSPACE`
    file.

2.  For each registered toolchain, Bazel performs the following checks:

    a. Does the toolchain match the requested toolchain type? If not, skip it.

    b. Do the toolchain's execution and target constraints match the constraints
       stated in the project's execution and target platforms? If yes, the
       toolchain is a match.

Because Bazel always selects the first matching toolchain, order the toolchains
by preference if you expect the possibility of multiple matches.

## Defining a toolchain

Defining a toolchain requires the following:

*  **Toolchain rule** - a rule invoked in a custom build or test rule that
   specifies the build tool configuration options particular to the toolchain
   and supported [platforms](platforms.html) (for example, [`go_toolchain`](https://github.com/bazelbuild/rules_go/blob/master/go/private/go_toolchain.bzl)).
   This rule must return a [`ToolchainInfo` provider](skylark/lib/platform_common.html#ToolchainInfo).
   The toolchain rule is lazily instantiated by Bazel on an as-needed basis.
   Because of this, a toolchain rule's dependencies can be as complex as needed,
   including reliance on remote repositories, without affecting builds that do
   not use them.

*  **Toolchain definition** - tells Bazel which [platform constraints](platforms.html#defining-a-platform)
   apply to the toolchain using the `toolchain()` rule. This rule must specify a
   unique toolchain type label, which is used as input during toolchain
   resolution.

*  **Toolchain registration** - makes the toolchain available to a Bazel project
   using the `register_toolchains()` function in the project's `WORKSPACE` file.

### Creating a toolchain rule

Toolchain rules are rules that create and return providers. To define a
toolchain rule, first determine the information that the new rule will require.

In the example below, we are adding support for a new programming language, so
we need to specify paths to the compiler and the system libraries, plus a flag
that determines the CPU architecture for which Bazel builds the output.

```python
def _my_toolchain_impl(ctx):
  toolchain = platform_common.ToolchainInfo(
    compiler = ctx.attr.compiler,
    system_lib = ctx.attr.system_lib,
    arch_flags = ctx.attr.arch_flags,
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

An example invocation of the rule looks as follows:

```python
my_toolchain(
  name = 'linux_toolchain_impl',
  compiler = '@remote_linux_repo//compiler:compiler_binary',
  system_lib = '@remote_linux_repo//library:system_library',
  arch_flags = [
    '--arch=Linux',
    '--debug_everything',
  ]
)

my_toolchain(
  name = 'darwin_toolchain_impl',
  compiler = '@remote_darwin_repo//compiler:compiler_binary',
  system_lib = '@remote_darwin_repo//library:system_library',
  arch_flags = [
    '--arch=Darwin',
    #'--debug_everything', # --debug_everything currently broken on Darwin
  ]
)
```

### Creating a toolchain definition

The toolchain definition is an instance of the `toolchain()` rule that specifies
the toolchain type, execution and target constraints, and the label of the
actual rule-specific toolchain. The use of the `toolchain()` rule enables the
lazy loading of toolchains.

Below is an example toolchain definition:

```python
toolchain_type(name = 'my_toolchain_type')

toolchain(
  name = 'linux_toolchain',
  toolchain_type = '//path/to:my_toolchain_type',
  exec_compatible_with = [
    '@bazel_tools//platforms:linux',
    '@bazel_tools//platforms:x86_64'],
  target_compatible_with = [
    '@bazel_tools//platforms:linux',
    '@bazel_tools//platforms:x86_64'],
  toolchain = ':linux_toolchain_impl',
)
```

### Registering a toolchain

Once the toolchain rule and definition exist, register the toolchain to make
Bazel aware of it. You can register a toolchain either via the project's
`WORKSPACE` file or specify it in the `--extra_toolchains` flag.

Below is an example toolchain registration in a `WORKSPACE` file:

```python
register_toolchains(
  '//path/to:linux_toolchain',
  '//path/to:darwin_toolchain',
)
```

## Using a toolchain in a rule

To use a toolchain in a rule, add the toolchain type to the rule
definition. For example:

```python
my_library = rule(
  ...
  toolchains = ['//path/to:my_toolchain_type']
  ...)
```

When using the `ctx.toolchains` rule, Bazel checks the execution and target
platforms, and select the first toolchain that matches. The rule implementation
can then access the toolchain as follows:

```python
def _my_library_impl(ctx):
  toolchain = ctx.toolchains['//path/to:my_toolchain_type']
  command = '%s -l %s %s' % (toolchain.compiler, toolchain.system_lib, toolchain.arch_flags)
  ...
```

## Debugging a toolchain

When adding toolchain support to an existing rule, use the
`--toolchain_resolution_debug` flag to make toolchain resolution verbose. Bazel
will output names of toolchains it is checking and skipping during the
resolution process.

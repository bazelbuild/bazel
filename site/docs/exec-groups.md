---
layout: documentation
title: Execution groups
---

# Execution groups


Execution groups allow for multiple execution platforms within a single target.
Each execution group has its own [toolchain](toolchains.md) dependencies and
performs its own [toolchain resolution](toolchains.md#toolchain-resolution).

## Current status

This feature is implemented but experimental. In order to use, you must set the
flag `--experimental_exec_group=true`.

## Background

Execution groups allow the rule author to define sets of actions, each with a
potentially different execution platform. Multiple execution platforms can allow
actions to execution differently, for example compiling an iOS app on a remote
(linux) worker and then linking/code signing on a local mac worker.

Being able to define groups of actions also helps alleviate the usage of action
mnemonics as a proxy for specifying actions. Mnemonics are not guaranteed to be
unique and can only reference a single action. This is especially helpful in
allocating extra resources to specific memory and processing intensive actions
like linking in c++ builds without over-allocating to less demanding tasks.

## Defining execution groups

During rule definition, rule authors can declare a set of execution groups. On
each execution group, the rule author can specify everything needed to select
an execution platform for that execution group, namely any constraints via
`exec_compatible_with` and toolchain types via `toolchain`. Execution groups do
not inherit the values of these
[parameters](https://docs.bazel.build/versions/master/skylark/lib/globals.html#rule)
from the rule to which they're attached.

TODO(juliexxia): link to exec_group method docs when they get released in bazel.

```python
# foo.bzl
my_rule = rule(
    _impl,
    exec_groups = {
        “link”: exec_group(
            exec_compatible_with = [ "@platforms//os:linux" ]
            toolchains = ["//foo:toolchain_type"],

        ),
        “test”: exec_group(
            toolchains = ["//foo_tools:toolchain_type"],
        ),
    },
    attrs = {
        "_compiler": attr.label(cfg = config.exec(“link”))
    },
)
```

In the code snippet above, you can see that tool dependencies can also specify
transition for an exec group using the
[`cfg`](https://docs.bazel.build/versions/master/skylark/lib/attr.html#label)
attribute param and the
[`config`](https://docs.bazel.build/versions/master/skylark/lib/config.html)
module. The module exposes an `exec` function which takes a single string
parameter which is the name of the exec group for which the dependency should be
built.

## Accessing execution groups

In the rule implementation, you can declare that actions should be run on the
execution platform of an execution group. You can do this by using the `exec_group`
param of action generating methods, specifically [`ctx.actions.run`]
(https://docs.bazel.build/versions/master/skylark/lib/actions.html#run) and
[`ctx.actions.run_shell`](https://docs.bazel.build/versions/master/skylark/lib/actions.html#run_shell).

```python
# foo.bzl
def _impl(ctx):
  ctx.actions.run(
     inputs = [ctx.attr._some_tool, ctx.srcs[0]]
     exec_group = "compile”,
     # ...
  )
```

Rule authors will also be able to access the [resolved toolchains]
(toolchains.md#toolchain-resolution) of execution groups, similarly to how you
can access the resolved toolchain of a target:

```python
# foo.bzl
def _impl(ctx):
  foo_info = ctx.exec_groups[‘link’].toolchains[‘//foo:toolchain_type”].fooinfo
  ctx.actions.run(
     inputs = [foo_info, ctx.srcs[0]]
     exec_group = "link”,
     # ...
  )
```

Note: If an action uses a toolchain from an execution group, but doesn't specify
that execution group in the action declaration, that may potentially cause
issues. A mismatch like this may not immediately cause failures, but is a latent
problem.

## Using execution groups to set execution properties

Execution groups are integrated with the
[`exec_properties`](be/common-definitions.html#common-attributes)
attribute that exists on every rule and allows the target writer to specify a
string dict of properties that is then passed to the execution machinery. For
example, if you wanted to set some property, say memory, for the target and give
certain actions a higher memory allocation, you would write an `exec_properties`
entry with an execution-group-augmented key, e.g.:

```python
# BUILD
my_rule(
    name = 'my_target',
    exec_properties = {
        'mem': '12G',
        'link.mem': '16G'
    }
    …
)
```

All actions with `exec_group = "link"` would see the exec properties
dictionary as `{"memory": "16G"}`. As you see here, execution-group-level
settings override target-level settings.


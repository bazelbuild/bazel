Project: /_project.yaml
Book: /_book.yaml

# Execution Groups

{% include "_buttons.html" %}

Execution groups allow for multiple execution platforms within a single target.
Each execution group has its own [toolchain](/extending/toolchains) dependencies and
performs its own [toolchain resolution](/extending/toolchains#toolchain-resolution).

## Background {:#background}

Execution groups allow the rule author to define sets of actions, each with a
potentially different execution platform. Multiple execution platforms can allow
actions to execution differently, for example compiling an iOS app on a remote
(linux) worker and then linking/code signing on a local mac worker.

Being able to define groups of actions also helps alleviate the usage of action
mnemonics as a proxy for specifying actions. Mnemonics are not guaranteed to be
unique and can only reference a single action. This is especially helpful in
allocating extra resources to specific memory and processing intensive actions
like linking in C++ builds without over-allocating to less demanding tasks.

## Defining execution groups {:#defining-exec-groups}

During rule definition, rule authors can
[declare](/rules/lib/globals/bzl#exec_group)
a set of execution groups. On each execution group, the rule author can specify
everything needed to select an execution platform for that execution group,
namely any constraints via `exec_compatible_with` and toolchain types via
`toolchain`.

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
        "_compiler": attr.label(cfg = config.exec("link"))
    },
)
```

In the code snippet above, you can see that tool dependencies can also specify
transition for an exec group using the
[`cfg`](/rules/lib/toplevel/attr#label)
attribute param and the
[`config`](/rules/lib/toplevel/config)
module. The module exposes an `exec` function which takes a single string
parameter which is the name of the exec group for which the dependency should be
built.

As on native rules, the `test` execution group is present by default on Starlark
test rules.

## Accessing execution groups {:#accessing-exec-groups}

In the rule implementation, you can declare that actions should be run on the
execution platform of an execution group. You can do this by using the `exec_group`
param of action generating methods, specifically [`ctx.actions.run`]
(/rules/lib/builtins/actions#run) and
[`ctx.actions.run_shell`](/rules/lib/builtins/actions#run_shell).

```python
# foo.bzl
def _impl(ctx):
  ctx.actions.run(
     inputs = [ctx.attr._some_tool, ctx.srcs[0]]
     exec_group = "compile",
     # ...
  )
```

Rule authors will also be able to access the [resolved toolchains](/extending/toolchains#toolchain-resolution)
of execution groups, similarly to how you
can access the resolved toolchain of a target:

```python
# foo.bzl
def _impl(ctx):
  foo_info = ctx.exec_groups["link"].toolchains["//foo:toolchain_type"].fooinfo
  ctx.actions.run(
     inputs = [foo_info, ctx.srcs[0]]
     exec_group = "link",
     # ...
  )
```

Note: If an action uses a toolchain from an execution group, but doesn't specify
that execution group in the action declaration, that may potentially cause
issues. A mismatch like this may not immediately cause failures, but is a latent
problem.

## Using execution groups to set execution properties {:#using-exec-groups-for-exec-properties}

Execution groups are integrated with the
[`exec_properties`](/reference/be/common-definitions#common-attributes)
attribute that exists on every rule and allows the target writer to specify a
string dict of properties that is then passed to the execution machinery. For
example, if you wanted to set some property, say memory, for the target and give
certain actions a higher memory allocation, you would write an `exec_properties`
entry with an execution-group-augmented key, such as:

```python
# BUILD
my_rule(
    name = 'my_target',
    exec_properties = {
        'mem': '12g',
        'link.mem': '16g'
    }
    …
)
```

All actions with `exec_group = "link"` would see the exec properties
dictionary as `{"mem": "16g"}`. As you see here, execution-group-level
settings override target-level settings.

### Execution groups for native rules {:#exec-groups-for-native-rules}

The following execution groups are available for actions defined by native rules:

* `test`: Test runner actions.
* `cpp_link`: C++ linking actions.

### Execution groups and platform execution properties {:#platform-execution-properties}

It is possible to define `exec_properties` for arbitrary execution groups on
platform targets (unlike `exec_properties` set directly on a target, where
properties for unknown execution groups are rejected). Targets then inherit the
execution platform's `exec_properties` that affect the default execution group
and any other relevant execution groups.

For example, suppose running a C++ test requires some resource to be available,
but it isn't required for compiling and linking; this can be modelled as
follows:

```python
constraint_setting(name = "resource")
constraint_value(name = "has_resource", constraint_setting = ":resource")

platform(
    name = "platform_with_resource",
    constraint_values = [":has_resource"],
    exec_properties = {
        "test.resource": "...",
    },
)

cc_test(
    name = "my_test",
    srcs = ["my_test.cc"],
    exec_compatible_with = [":has_resource"],
)
```

`exec_properties` defined directly on targets take precedence over those that
are inherited from the execution platform.

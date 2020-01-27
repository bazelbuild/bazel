# Common scalar build settings.

## Overview

These definitions are only for use in Bazel built-in tools.  In all
other cases, use flag type definitions from
github.com/bazelbuild/bazel-skylib/rules/common_settings.bzl.

<a name="basic-example"></a>
## Basic Example

### //tools/my_pkg/BUILD

```python
load("//tools/config:common_settings.bzl", "bool_flag")

bool_flag(
    name = "experimental_new_behavior",
    build_setting_default = False,
)
```

### //tools/my_pkg/myrule.bzl

```python
load("//tools/flags:flags.bzl", "BuildSettingInfo")


def _myrule_impl(ctx):
    if ctx.attr._experimental_new_behavior[BuildSettingInfo].value:
       ...
    ...


myrule = rule(
    implementation = _myrule_impl.
    attrs = {
        ...

        "_experimental_new_behavior": attr.label(
            default = "//tools/my_pkg:experimental_new_behavior"),
    },
)
```

---
layout: documentation
title: Skylark Repository Rules
---
# Repository Rules

**Status: Experimental**. We may make breaking changes to the API, but we will
  announce them and help you update your code.

An [external repository](/docs/external.md) is a rule that can be used only
in the `WORKSPACE` file and enable non-hermetic operation at the loading phase
of Bazel. Each external repository rule creates its own workspace, with its
own BUILD files and artifacts. They can be used to depend on third-party
libraries (such as Maven packaged libraries) but also to generate BUILD files
specific to the host Bazel is running on.

## Repository Rule creation

In a Skylark extension, use the
[repository_rule](lib/globals.html#repository_rule) function to create a new
repository rule and store it in a global variable.

A custom repository rule can be used just like a native repository rule. It
has a mandatory `name` attribute and every target present in its build files
can be refered as `@<name>//package:target` where `<name>` is the value of the
`name` attribute.

The rule is loaded when you explictly build it, or if it is a dependency of
the build. In this case, Bazel will execute its `implementation` function. This
function describe how to creates the repository, its content and BUILD files.

## Attributes

An attribute is a rule argument, such as `url` or `sha256`. You must list
the attributes and their types when you define a repository rule.

```python
local_repository = repository_rule(
    implementation=_impl,
    local=True,
    attrs={"path": attr.string(mandatory=True)})
```

`name` attributes are implicitely defined for all `repository_rule`s.
To access an attribute, use `repository_ctx.attr.<attribute_name>`.
The name of a repository rule is accessible with `repository_ctx.name`.

If an attribute name starts with `_` it is private and users cannot set it.

## Implementation function

Every repository rule requires an `implementation` function. It contains the
actual logic of the rule and is executed strictly in the Loading Phase.
The function has exactly one input parameter, `repository_ctx`, and should
always returns `None`. The input parameter `repository_ctx` can be used to
access attribute values, and non-hermetic functions (finding a binary,
exuting a binary, creating a file in the repository or downloading a file
from the Internet). See [the library](lib/repository_ctx.html) for more
context. Example:

```python
def _impl(repository_ctx):
  repository_ctx.symlink(repository_ctx.attr.path, "")

local_repository = repository_rule(
    implementation=_impl,
    ...)
```

## Examples

For now, we only have one full example of usage of the `repository_rule`:
[C++ auto-configured toolchain](https://github.com/bazelbuild/bazel/blob/9116b3e99af2fd31d92c9bb7c37905a1675456c1/tools/cpp/cc_configure.bzl#L288).

This example uses a Skylark repository rule to automatically create the
C++ configuration files for Bazel by looking for the local C++ compiler, the
environment and the flags the C++ compiler supports.


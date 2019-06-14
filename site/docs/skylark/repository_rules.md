---
layout: documentation
title: Repository Rules
---
# Repository Rules

**Status: Experimental**. We may make breaking changes to the API, but we will
  announce them.

An [external repository](../external.md) is a rule that can be used only
in the `WORKSPACE` file and enables non-hermetic operation at the loading phase
of Bazel. Each external repository rule creates its own workspace, with its
own BUILD files and artifacts. They can be used to depend on third-party
libraries (such as Maven packaged libraries) but also to generate BUILD files
specific to the host Bazel is running on.

## Repository Rule creation

In a `.bzl` file, use the
[repository_rule](lib/globals.html#repository_rule) function to create a new
repository rule and store it in a global variable.

A custom repository rule can be used just like a native repository rule. It
has a mandatory `name` attribute and every target present in its build files
can be referred as `@<name>//package:target` where `<name>` is the value of the
`name` attribute.

The rule is loaded when you explicitly build it, or if it is a dependency of
the build. In this case, Bazel will execute its `implementation` function. This
function describe how to create the repository, its content and BUILD files.

## Attributes

An attribute is a rule argument, such as `url` or `sha256`. You must list
the attributes and their types when you define a repository rule.

```python
local_repository = repository_rule(
    implementation=_impl,
    local=True,
    attrs={"path": attr.string(mandatory=True)})
```

`name` attributes are implicitly defined for all `repository_rule`s.
To access an attribute, use `repository_ctx.attr.<attribute_name>`.
The name of a repository rule is accessible with `repository_ctx.name`.

If an attribute name starts with `_` it is private and users cannot set it.

## Implementation function

Every repository rule requires an `implementation` function. It contains the
actual logic of the rule and is executed strictly in the Loading Phase.

The function has exactly one input parameter, `repository_ctx`. The function
returns either `None` to signify that the rule is reproducible, or a dict with a
set of parameters for that rule that would turn that rule into a reproducible
one generating the same repository. For example, for a rule tracking a git
repository that would mean returning a specific commit identifier instead of a
floating branch that was originally specified.

The input parameter `repository_ctx` can be used to
access attribute values, and non-hermetic functions (finding a binary,
executing a binary, creating a file in the repository or downloading a file
from the Internet). See [the library](lib/repository_ctx.html) for more
context. Example:

```python
def _impl(repository_ctx):
  repository_ctx.symlink(repository_ctx.attr.path, "")

local_repository = repository_rule(
    implementation=_impl,
    ...)
```

## When is the implementation function executed?

If the repository is declared as `local` then change in a dependency
in the dependency graph (including the WORKSPACE file itself) will
cause an execution of the implementation function.

The implementation function can be _restarted_ if a dependency it
request is _missing_. The beginning of the implementation function
will be re-executed after the dependency has been resolved. To avoid
unnecessary restarts (which are expensive, as network access might
have to be repeated), label arguments are prefetched, provided all
label arguments can be resolved to an existing file. Note that resolving
a path from a string or a label that was constructed only during execution
of the function might still cause a restart.

Finally, for non-`local` repositories, only a change in the following
dependencies might cause a restart:

- `.bzl` files needed to define the repository rule.
- Declaration of the repository rule in the `WORKSPACE` file.
- Value of any environment variable declared with the `environ`
attribute of the
[`repository_rule`](https://docs.bazel.build/skylark/lib/globals.html#repository_rule)
function. The value of those environment variable can be enforced from
the command line with the
[`--action_env`](https://docs.bazel.build/command-line-reference.html#flag--action_env)
flag (but this flag will invalidate every action of the build).
- Content of any file used and referred to by a label (e.g.,
  `//mypkg:label.txt` not `mypkg/label.txt`).

## Examples

- [C++ auto-configured toolchain](https://github.com/bazelbuild/bazel/blob/ac29b78000afdb95afc7e97efd2b1299ebea4dac/tools/cpp/cc_configure.bzl#L288):
it uses a repository rule to automatically create the
C++ configuration files for Bazel by looking for the local C++ compiler, the
environment and the flags the C++ compiler supports.

- [Go repositories](https://github.com/bazelbuild/rules_go/blob/67bc217b6210a0922d76d252472b87e9a6118fdf/go/private/go_repositories.bzl#L195)
  uses several `repository_rule` to defines the list of dependencies
  needed to use the Go rules.

- [rules_jvm_external](https://github.com/bazelbuild/rules_jvm_external) creates
  an external repository called `@maven` by default that generates build targets
  for every Maven artifact in the transitive dependency tree.

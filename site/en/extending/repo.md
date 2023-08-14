Project: /_project.yaml
Book: /_book.yaml

# Repository Rules

{% include "_buttons.html" %}

This page covers how to create repository rules and provides examples for
more details.

An [external repository](/docs/external) is a rule that can be used only
in the `WORKSPACE` file and enables non-hermetic operation at the loading phase
of Bazel. Each external repository rule creates its own workspace, with its
own `BUILD` files and artifacts. They can be used to depend on third-party
libraries (such as Maven packaged libraries) but also to generate `BUILD` files
specific to the host Bazel is running on.

## Repository rule creation

In a `.bzl` file, use the
[repository_rule](/rules/lib/globals/bzl#repository_rule) function to create a new
repository rule and store it in a global variable.

A custom repository rule can be used just like a native repository rule. It
has a mandatory `name` attribute and every target present in its build files
can be referred as `@<name>//package:target` where `<name>` is the value of the
`name` attribute.

The rule is loaded when you explicitly build it, or if it is a dependency of
the build. In this case, Bazel will execute its `implementation` function. This
function describe how to create the repository, its content and `BUILD` files.

## Attributes

An attribute is a rule argument, such as `url` or `sha256`. You must list
the attributes and their types when you define a repository rule.

```python
local_repository = repository_rule(
    implementation=_impl,
    local=True,
    attrs={"path": attr.string(mandatory=True)})
```

To access an attribute, use `repository_ctx.attr.<attribute_name>`.

All `repository_rule`s have implicitly defined attributes (just like build
rules). The two implicit attributes are `name` (just like for build rules) and
`repo_mapping`. The name of a repository rule is accessible with
`repository_ctx.name`. The meaning of `repo_mapping` is the same as for the
native repository rules
[`local_repository`](https://bazel.build/reference/be/workspace#local_repository.repo_mapping)
and
[`new_local_repository`](https://bazel.build/reference/be/workspace#new_local_repository.repo_mapping).

If an attribute name starts with `_` it is private and users cannot set it.

## Implementation function

Every repository rule requires an `implementation` function. It contains the
actual logic of the rule and is executed strictly in the Loading Phase.

The function has exactly one input parameter, `repository_ctx`. The function
returns either `None` to signify that the rule is reproducible given the
specified parameters, or a dict with a set of parameters for that rule that
would turn that rule into a reproducible one generating the same repository. For
example, for a rule tracking a git repository that would mean returning a
specific commit identifier instead of a floating branch that was originally
specified.

The input parameter `repository_ctx` can be used to
access attribute values, and non-hermetic functions (finding a binary,
executing a binary, creating a file in the repository or downloading a file
from the Internet). See [the library](/rules/lib/builtins/repository_ctx) for more
context. Example:

```python
def _impl(repository_ctx):
  repository_ctx.symlink(repository_ctx.attr.path, "")

local_repository = repository_rule(
    implementation=_impl,
    ...)
```

## When is the implementation function executed?

The implementation function of a repository is executed when Bazel needs a
target from that repository, for example when another target (in another
repository) depends on it or if it is mentioned on the commmand line. The
implementation function is then expected to create the repository in the file
system. This is called "fetching" the repository.

In contrast to regular targets, repositories are not necessarily re-fetched when
something changes that would cause the repository to be different. This is
because there are things that Bazel either cannot detect changes to or it would
cause too much overhead on every build (for example, things that are fetched
from the network). Therefore, repositories are re-fetched only if one of the
following things changes:

* The parameters passed to the declaration of the repository in the
  `WORKSPACE` file.
* The Starlark code comprising the implementation of the repository.
* The value of any environment variable declared with the `environ`
  attribute of the [`repository_rule`](/rules/lib/globals/bzl#repository_rule).
  The values of these environment variables can be hard-wired on the command
  line with the
  [`--action_env`](/reference/command-line-reference#flag--action_env)
  flag (but this flag will invalidate every action of the build).
* The content of any file passed to the `read()`, `execute()` and similar
  methods of `repository_ctx` which is referred to by a label (for example,
  `//mypkg:label.txt` but not `mypkg/label.txt`)
* When `bazel sync` is executed.

There are two parameters of `repository_rule` that control when the repositories
are re-fetched:

* If the `configure` flag is set, the repository is only re-fetched on
  `bazel sync` when the` --configure` parameter is passed to it (if the
  attribute is unset, this command will not cause a re-fetch)
* If the `local` flag is set, in addition to the above cases, the repository is
  also re-fetched when the Bazel server restarts or when any file that affects
  the declaration of the repository changes (e.g. the `WORKSPACE` file or a file
  it loads), regardless of whether the changes resulted in a change to the
  declaration of the repository or its code.

  Non-local repositories are not re-fetched in these cases. This is because
  these repositories are assumed to talk to the network or be otherwise
  expensive.

## Restarting the implementation function

The implementation function can be _restarted_ while a repository is being
fetched if a dependency it requests is _missing_. In that case, the execution of
the implementation function will stop, the missing dependency is resolved and
the function will be re-executed after the dependency has been resolved. To
avoid unnecessary restarts (which are expensive, as network access might
have to be repeated), label arguments are prefetched, provided all
label arguments can be resolved to an existing file. Note that resolving
a path from a string or a label that was constructed only during execution
of the function might still cause a restart.

## Forcing refetch of external repositories

Sometimes, an external repository can become outdated without any change to its
definition or dependencies. For example, a repository fetching sources might
follow a particular branch of a third-party repository, and new commits are
available on that branch. In this case, you can ask bazel to refetch all
external repositories unconditionally by calling `bazel sync`.

Moreover, some rules inspect the local machine and might become
outdated if the local machine was upgraded. Here you can ask bazel to
only refetch those external repositories where the
[`repository_rule`](/rules/lib/globals#repository_rule)
definition has the `configure` attribute set, use `bazel sync --configure`.


## Examples

- [C++ auto-configured toolchain](https://cs.opensource.google/bazel/bazel/+/master:tools/cpp/cc_configure.bzl;drc=644b7d41748e09eff9e47cbab2be2263bb71f29a;l=176):
it uses a repository rule to automatically create the
C++ configuration files for Bazel by looking for the local C++ compiler, the
environment and the flags the C++ compiler supports.

- [Go repositories](https://github.com/bazelbuild/rules_go/blob/67bc217b6210a0922d76d252472b87e9a6118fdf/go/private/go_repositories.bzl#L195)
  uses several `repository_rule` to defines the list of dependencies
  needed to use the Go rules.

- [rules_jvm_external](https://github.com/bazelbuild/rules_jvm_external) creates
  an external repository called `@maven` by default that generates build targets
  for every Maven artifact in the transitive dependency tree.

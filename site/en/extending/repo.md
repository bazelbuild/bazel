Project: /\_project.yaml
Book: /\_book.yaml

# Repository Rules

{% include "_buttons.html" %}

This page covers how to define repository rules and provides examples for more
details.

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

Attribute are rule arguments passed as a dict to the `attrs` rule argument.
The attributes and their types are defined are listed when you define a
repository rule. An example definining `url` and `sha256` attributes as
strings:

```python
local_repository = repository_rule(
    implementation=_impl,
    local=True,
    attrs={
        "url": attr.string(mandatory=True)
        "sha256": attr.string(mandatory=True)
    }
)
```

To access an attribute within the implementation function, use
`repository_ctx.attr.<attribute_name>`:

```python
def _impl(repository_ctx):
    url = repository_ctx.attr.url
    checksum = repository_ctx.attr.sha256
```

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

Every repo rule requires an `implementation` function. It contains the actual
logic of the rule and is executed strictly in the Loading Phase.

The function has exactly one input parameter, `repository_ctx`. The function
returns either `None` to signify that the rule is reproducible given the
specified parameters, or a dict with a set of parameters for that rule that
would turn that rule into a reproducible one generating the same repository. For
example, for a rule tracking a git repository that would mean returning a
specific commit identifier instead of a floating branch that was originally
specified.

The input parameter `repository_ctx` can be used to access attribute values, and
non-hermetic functions (finding a binary, executing a binary, creating a file in
the repository or downloading a file from the Internet). See [the API
docs](/rules/lib/builtins/repository_ctx) for more context. Example:

```python
def _impl(repository_ctx):
  repository_ctx.symlink(repository_ctx.attr.path, "")

local_repository = repository_rule(
    implementation=_impl,
    ...)
```

## When is the implementation function executed?

The implementation function of a repo rule is executed when Bazel needs a target
from that repository, for example when another target (in another repo) depends
on it or if it is mentioned on the command line. The implementation function is
then expected to create the repo in the file system. This is called "fetching"
the repo.

In contrast to regular targets, repos are not necessarily re-fetched when
something changes that would cause the repo to be different. This is because
there are things that Bazel either cannot detect changes to or it would cause
too much overhead on every build (for example, things that are fetched from the
network). Therefore, repos are re-fetched only if one of the following things
changes:

- The attributes passed to the repo rule invocation.
- The Starlark code comprising the implementation of the repo rule.
- The value of any environment variable passed to `repository_ctx`'s
  `getenv()` method or declared with the `environ` attribute of the
  [`repository_rule`](/rules/lib/globals/bzl#repository_rule). The values of
  these environment variables can be hard-wired on the command line with the
  [`--repo_env`](/reference/command-line-reference#flag--repo_env) flag.
- The existence, contents, and type of any paths being
  [`watch`ed](/rules/lib/builtins/repository_ctx#watch) in the implementation
  function of the repo rule.
  - Certain other methods of `repository_ctx` with a `watch` parameter, such
    as `read()`, `execute()`, and `extract()`, can also cause paths to be
    watched.
  - Similarly, [`repository_ctx.watch_tree`](/rules/lib/builtins/repository_ctx#watch_tree)
    and [`path.readdir`](/rules/lib/builtins/path#readdir) can cause paths
    to be watched in other ways.
- When `bazel fetch --force` is executed.

There are two parameters of `repository_rule` that control when the repositories
are re-fetched:

- If the `configure` flag is set, the repository is only re-fetched on `bazel
fetch` when the` --configure` parameter is passed to it (if the attribute is
  unset, this command will not cause a re-fetch)
- If the `local` flag is set, in addition to the above cases, the repo is also
  re-fetched when the Bazel server restarts.

## Forcing refetch of external repositories

Sometimes, an external repo can become outdated without any change to its
definition or dependencies. For example, a repo fetching sources might follow a
particular branch of a third-party repository, and new commits are available on
that branch. In this case, you can ask bazel to refetch all external repos
unconditionally by calling `bazel fetch --force --all`.

Moreover, some repo rules inspect the local machine and might become outdated if
the local machine was upgraded. Here you can ask Bazel to only refetch those
external repos where the [`repository_rule`](/rules/lib/globals#repository_rule)
definition has the `configure` attribute set, use `bazel fetch --all
--configure`.

## Examples

- [C++ auto-configured
  toolchain](https://cs.opensource.google/bazel/bazel/+/master:tools/cpp/cc_configure.bzl;drc=644b7d41748e09eff9e47cbab2be2263bb71f29a;l=176):
  it uses a repo rule to automatically create the C++ configuration files for
  Bazel by looking for the local C++ compiler, the environment and the flags
  the C++ compiler supports.

- [Go repositories](https://github.com/bazelbuild/rules_go/blob/67bc217b6210a0922d76d252472b87e9a6118fdf/go/private/go_repositories.bzl#L195)
  uses several `repository_rule` to defines the list of dependencies needed to
  use the Go rules.

- [rules_jvm_external](https://github.com/bazelbuild/rules_jvm_external)
  creates an external repository called `@maven` by default that generates
  build targets for every Maven artifact in the transitive dependency tree.

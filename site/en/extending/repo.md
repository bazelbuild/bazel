Project: /_project.yaml
Book: /_book.yaml

# Repository Rules

{% include "_buttons.html" %}

This page covers how to define repository rules and provides examples for more
details.

An [external repository](/external/overview#repository) is a directory tree,
containing source files usable in a Bazel build, which is generated on demand by
running its corresponding **repo rule**. Repos can be defined in a multitude of
ways, but ultimately, each repo is defined by invoking a repo rule, just as
build targets are defined by invoking build rules. They can be used to depend on
third-party libraries (such as Maven packaged libraries) but also to generate
`BUILD` files specific to the host Bazel is running on.

## Repository rule definition

In a `.bzl` file, use the
[repository_rule](/rules/lib/globals/bzl#repository_rule) function to define a
new repo rule and store it in a global variable. After a repo rule is defined,
it can be invoked as a function to define repos. This invocation is usually
performed from inside a [module extension](/external/extension) implementation
function.

The two major components of a repo rule definition are its attribute schema and
implementation function. The attribute schema determines the names and types of
attributes passed to a repo rule invocation, and the implementation function is
run when the repo needs to be fetched.

## Attributes

Attributes are arguments passed to the repo rule invocation. The schema of
attributes accepted by a repo rule is specified using the `attrs` argument when
the repo rule is defined with a call to `repository_rule`. An example defining
`url` and `sha256` attributes as strings:

```python
http_archive = repository_rule(
    implementation=_impl,
    attrs={
        "url": attr.string(mandatory=True),
        "sha256": attr.string(mandatory=True),
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

All `repository_rule`s have the implicitly defined attribute `name`. This is a
string attribute that behaves somewhat magically: when specified as an input to
a repo rule invocation, it takes an apparent repo name; but when read from the
repo rule's implementation function using `repository_ctx.attr.name`, it returns
the canonical repo name.

## Implementation function

Every repo rule requires an `implementation` function. It contains the actual
logic of the rule and is executed strictly in the Loading Phase.

The function has exactly one input parameter, `repository_ctx`. The function
returns either `None` to signify that the rule is reproducible given the
specified parameters, or a dict with a set of parameters for that rule that
would turn that rule into a reproducible one generating the same repo. For
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

*   The attributes passed to the repo rule invocation.
*   The Starlark code comprising the implementation of the repo rule.
*   The value of any environment variable passed to `repository_ctx`'s
    `getenv()` method or declared with the `environ` attribute of the
    [`repository_rule`](/rules/lib/globals/bzl#repository_rule). The values of
    these environment variables can be hard-wired on the command line with the
    [`--repo_env`](/reference/command-line-reference#flag--repo_env) flag.
*   The existence, contents, and type of any paths being
    [`watch`ed](/rules/lib/builtins/repository_ctx#watch) in the implementation
    function of the repo rule.
    *   Certain other methods of `repository_ctx` with a `watch` parameter, such
        as `read()`, `execute()`, and `extract()`, can also cause paths to be
        watched.
    *   Similarly, [`repository_ctx.watch_tree`](/rules/lib/builtins/repository_ctx#watch_tree)
        and [`path.readdir`](/rules/lib/builtins/path#readdir) can cause paths
        to be watched in other ways.
*   When `bazel fetch --force` is executed.

There are two parameters of `repository_rule` that control when the repositories
are re-fetched:

*   If the `configure` flag is set, the repository is only re-fetched on `bazel
    fetch` when the` --configure` parameter is passed to it (if the attribute is
    unset, this command will not cause a re-fetch)
*   If the `local` flag is set, in addition to the above cases, the repo is also
    re-fetched when the Bazel server restarts.

## Forcing refetch of external repos

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

-   [C++ auto-configured
    toolchain](https://cs.opensource.google/bazel/bazel/+/master:tools/cpp/cc_configure.bzl;drc=644b7d41748e09eff9e47cbab2be2263bb71f29a;l=176):
    it uses a repo rule to automatically create the C++ configuration files for
    Bazel by looking for the local C++ compiler, the environment and the flags
    the C++ compiler supports.

-   [Go repositories](https://github.com/bazelbuild/rules_go/blob/67bc217b6210a0922d76d252472b87e9a6118fdf/go/private/go_repositories.bzl#L195)
    uses several `repository_rule` to defines the list of dependencies needed to
    use the Go rules.

-   [rules_jvm_external](https://github.com/bazelbuild/rules_jvm_external)
    creates an external repository called `@maven` by default that generates
    build targets for every Maven artifact in the transitive dependency tree.
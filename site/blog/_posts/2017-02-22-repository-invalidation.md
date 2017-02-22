---
layout: posts
title: Invalidation of repository rules
---

[Remote repositories](/docs/external.html) are the way to use dependencies from
"outside" of the Bazel world in Bazel. Using them, you can download binaries from the
internet or use some from your own host. You can even use
[Skylark](/skylark/repository_rules.html) to define your own repository rules to depend
on a custom package manager or to implement
[auto-configuration rules](/blog/2016/03/31/autoconfiguration.html).

This post explains when Skylark repositories are invalidated and hence when they are executed.

## Dependencies

The implementation attribute of the
[`repository_rule`](https://bazel.build/versions/master/docs/skylark/lib/globals.html#repository_rule)
defines a function (the _fetch_ operation) that is executed inside a
[Skyframe function](/designs/skyframe.html). This function is executed when
one of its dependencies change.

For repository that are declared `local` (set `local = True` in the call to the
`repository_rule` function), the _fetch_ operation is performed on every call of the
Skyframe function.

Since a lot of dependencies can trigger this execution (if any part of the `WORKSPACE`
file change for instance), a supplemental mechanism ensure that we re-execute the
_fetch_ operation only when stricly needed for non-`local` repository rules (see the
[design doc](/designs/2016/10/18/repository-invalidation.html) for more details).

After [cr.bazel.build/8218](https://cr.bazel.build/8218) is released, Bazel will
re-perform the `fetch` operation if and only if any of the following
dependencies change:

- Skylark files needed to define the repository rule.
- Declaration of the repository rule in the `WORKSPACE` file.
- Value of any environment variable declared with the `environ` attribute of the [`repository_rule`](https://bazel.build/versions/master/docs/skylark/lib/globals.html#repository_rule) function. The value of those environment variable can be enforced from the command line with the
[`--action_env`](/docs/command-line-reference.html#flag--action_env) flag (but this
flag will invalidate every action of the build).
- Content of any file used and referred using a label (e.g., `//mypkg:label.txt` not `mypkg/label.txt`).

## Good practices regarding refetching

### Declare your repository as local very carefully

First and foremost, declaring a repository `local` should be done only for rule that
needs to be eagerly invalidated and are fast to update. For native rule, this is used only
for [`local_repository`](/docs/be/workspace.html#local_repository) and
[`new_local_repository`](/docs/be/workspace.html#new_local_repository).

### Put all slow operation at the end, resolve dependencies first

Since a dependency might be unresolved when asked for, the function will be executed
up to where the dependency is requested and all that part will be replayed if the
dependency is not resolved. Put those file dependencies at the top, for instance prefer

```python
def _impl(repository_ctx):
   repository_ctx.file("BUILD", repository_ctx.attr.build_file)
   repository_ctx.download("BIGFILE", sha256 = "...")

myrepo = repository_rule(_impl, attrs = {"build_file": attr.label()})
```

over

```python
def _impl(repository_ctx):
   repository_ctx.download("BIGFILE")
   repository_ctx.file("BUILD", repository_ctx.attr.build_file)

myrepo = repository_rule(_impl, attrs = {"build_file": attr.label()})
```

(in the later example, the download operation will be re-executed if `build_file` is not
resolved when executing the `fetch` operation).

### Declare your environment variables

To avoid spurious refetch of repository rules (and the impossibility of tracking all
usages of environmnent variables), only environment variables that have been declared
through the `environ` attribute of the `repository_rule` function are invalidating
the repositories.

Therefore, if you think you should re-run if an environment variable changes (like
for auto-configuration rules), you should declare those dependencies, or your user
will have to do `bazel clean --expunge` each time they change their environment.

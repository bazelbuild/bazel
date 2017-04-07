---
layout: contribute
title: Invalidation of remote repositories
---

# Design Document: Invalidation of remote repositories

**Design documents are not descriptions of the current functionality of Bazel.
Always go to the documentation for current information.**


**Status**: Implemented

**Author**: [Damien Martin-Guillerez](dmarting@google.com)

**Design document published**: 18 October 2016

## State at commit [808a651](https://github.com/bazelbuild/bazel/commit/808a6518519501cfd32755a229d5dddf70e33557)

[Remote repositories](/docs/external.html) are fetched the first
time a build that depends on a repository is launched. The next
time the same build happens, the already fetched repositories
are not refetched, saving on download times or other expensive
operations.

This behavior is also enforced even when the Bazel server
is restarted by serializing the repository rule in the workspace
file. A file named `@<repositoryName>.marker` is
[created](https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/rules/repository/RepositoryDelegatorFunction.java#L131)
for each repository with a
[fingerprint of the serialized rule](https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/rules/repository/RepositoryDelegatorFunction.java#L192). On
next fetch, if that fingerprint has not changed, the rule is not
refetched. This is not applied if the repository rule is marked
as
[`local`](https://www.bazel.io/versions/master/docs/skylark/lib/globals.html#repository_rule)
because fetching a local repository is assumed to be
[fast](https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/rules/repository/RepositoryDelegatorFunction.java#L125).

## Shortcomings

These consideration were well-suited when the implementation of
repository rules were not depending on Skylark file. With the introduction of
[Skylark repositories](https://www.bazel.io/versions/master/docs/skylark/repository_rules.html),
several issues appeared:

- [Change in the skylark implementation of the rule does not
  trigger a refetch of the rule](https://github.com/bazelbuild/bazel/issues/1022),
  nor does a change in one of the template files that relies on that
  rule: the rule marker does not contains this information.
- There is [no way to re-configure a repository used for
  auto-configuration](https://github.com/bazelbuild/bazel/issues/974),
  leading to
  [excessive uses of `bazel clean --expunge`](https://github.com/tensorflow/tensorflow/blob/60d54d6b8524bcaf512f53384b307fae47b953d2/configure#L25).
- The invalidation behavior of repository rules are unclear and
  difficult to explain.

## Proposed solution

### Invalidation on the environment

Right now rules are not invalidated on the environment:

- Invalidation on accessing
  [`repository_ctx.os.environ`](https://www.bazel.io/versions/master/docs/skylark/lib/repository_os.html#environ)
  would generate invalidation on environment variable that might be
  volatile (e.g. `CC` when you want to use one C++ compiler and you
  reset your environment) and might miss other environment variables
  due to computed variable names.
- There is no way to represent environment variables that influence
  [`repository_ctx.execute`](https://www.bazel.io/versions/master/docs/skylark/lib/repository_ctx.html#execute).

This document proposes to add a way to declare a dependency on an
environment variable value that would trigger a refetch of a
repository. An optional attribute `environ` would be added to the
[`repository_rule`](https://www.bazel.io/versions/master/docs/skylark/lib/globals.html#repository_rule)
method, taking a list of strings and would trigger invalidation of the
repository on any of change to those environment variables. E.g.:

```python
my_repo = repository_rule(impl = _impl, environ = ["FOO", "BAR"])
```

`my_repo` would be refetched on any change to the environment
variables `FOO` or `BAR` but not if the environment variable `BAZ`
would changes.

To be consistent with the
[new environment specification](https://www.bazel.io/designs/2016/06/21/environment.html)
mechanism, the environment available through
[`repository_ctx.os.environ`](https://www.bazel.io/versions/master/docs/skylark/lib/repository_os.html#environ)
or transmitted to
[`repository_ctx.execute`](https://www.bazel.io/versions/master/docs/skylark/lib/repository_ctx.html#execute)
will take values from the `--action_env` flag, when specified. I.e. if
`--action_env FOO=BAR --action_env BAR` are specified, and the
environment set `FOO=BAZ`, `BAR=FOO`, `BAZ=BAR`, then the actual
`repository_ctx.os.environ` map would contain `{"FOO": "BAR", "BAR":
"FOO", "BAZ": "BAR" }`. This would ensure that the environment seen by
repository rules is consistent with the one seen by actions (a
repository rule see more than an action, leaving the rule
writer the ability to filter the environment more finely).

Both these changes should allow Bazel to do auto-configuration
based on environment variables:

- Setting some environment variables would actually retrigger
  auto-configuration, corresponding to how the rule writter designed
  it (and not based on some assumption from Bazel).
- The user set specific environment variables through the `--action_env`
  flag, and fix this environment using `bazel info client-env`.

### Serialization of Skyframe dependencies

A `local` rule will be invalidated when any of its skyframe
dependencies change. For non-`local` rule, a marker file
will be stored on the external directory with a summary of the
dependencies of the rule. At each fetch operation, we check
the existence of the marker file and verify each dependency.
If one of them have changed, we would refetch that repository.

To avoid unnecessary re-download of artifacts, a content-addressable
cache has been developed for downloads (and thus not discuted here).

The marker file will be a manifest containing the following
items:

- A fingerprint of the serialized rule and the rule specific data
  (e.g., maven server information for `maven_jar`).
- The declared environment (list of name, value pairs) through the
  `environ` attribute of the repository rule.
- The list of `FileValue`-s requested by
  [`getPathFromLabel`](https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/bazel/repository/skylark/SkylarkRepositoryContext.java#L613)
  and the corresponding file content digest.
- The transtive hash of the `Extension` definining the repository rule.
  This transitive hash is computed from the hash of the current extension
  and the extension loaded from it. This means that a repository function
  will get invalidated as soon as the extension file content changes, which
  is an over invalidation. However, getting an optimal result would require
  correct serialization of Skylark extensions.

## Implementation plan

1. Modify the `SkylarkRepositoryFunction#getClientEnvironment` method
   to get the values from the `--action_env` flag.
2. Adds a `markerData` map argument to `RepositoryFunction#fetch` so
   `SkylarkRepositoryFunction` can include those change. This attribute
   should be mutable so a repository can add more data to be stored
   in the marker file. Adds a corresponding function for
   verification, `verifyMarkerManifest`, that would take a marker data
   map and return a tri-state: true if the repository is up to date,
   false if it needs refetch and null if additional Skyframe dependency
   need to be resolved for answering.
3. Add the `environ` attribute to the `repository_rule` function and
   the dependency on the Skyframe values for the environment. Also create
   a `SkyFunction` for processed environment after the `--action_env`
   flag.
4. Adds the `environ` values to the marker file through the
   `getMarkerManifest` function.
5. Adds the `FileValue`-s to the marker file, adding all the files
   requested through the `getPath` method to a specific builder that
   will be passed to the `SkylarkRepositoryContext`.
6. Adds the extension to the marker file by passing the
   `transitiveHashCode` of the Skylark `Environment` to the marker
   manifest.

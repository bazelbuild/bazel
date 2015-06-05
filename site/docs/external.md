---
layout: documentation
---

# Working with external dependencies

Bazel is designed to have absolutely everything needed for a build, from source
code to libraries to compilers, under one directory (the build root). This is
impractical for some version control systems and goes against how many existing
projects are structured. Thus, Bazel has a system for pulling in dependencies
from outside of the build root.

External dependencies can be specified in a _WORKSPACE_ file in the build root.
This _WORKSPACE_ file uses the same Python-like syntax of BUILD files, but
allows a different set of rules.

## Fetching dependencies

By default, external dependencies are fetched as needed during `bazel build`. If
you would like to disable this behavior or prefetch dependencies, use
[`bazel fetch`](http://bazel.io/docs/bazel-user-manual.html#fetch).

## Transitive dependencies

Bazel only reads dependencies listed in your build root's _WORKSPACE_ file. This
means that if your project (_A_) depends on another project (_B_) which list a
dependency on project _C_ in its _WORKSPACE_ file, you'll have to add both _B_
and _C_ to your project's _WORKSPACE_ file. This can balloon the _WORKSPACE_
file size, but hopefully limits the chances of having one library include _C_
at version 1.0 and another include _C_ at 2.0.

# Converting existing projects

To convert a Maven project, first run the `generate_workspace` tool:

```bash
$ bazel run src/main/java/com/google/devtools/build/workspace:generate_workspace /path/to/your/maven/project >> WORKSPACE
```

This will parse the _pom.xml_ file and discover project dependencies. All of
these dependencies will be written in
[`maven_jar`](http://bazel.io/docs/build-encyclopedia.html#maven_jar) format to
stdout, which can be redirected or copied to the _WORKSPACE_ file.

At the moment, `generate_workspace` will only include direct dependencies.

You will still need to manually add these libraries as dependencies of your
`java_` targets.

# Types of external dependencies

There are a few basic types of external dependencies that can be created.

## Combining Bazel projects

If you have a second Bazel project that you'd like to use targets from, you can
use
[`local_repository`](http://bazel.io/docs/build-encyclopedia.html#local_repository)
or [`http_archive`](http://bazel.io/docs/build-encyclopedia.html#http_archive)
to symlink it from the local filesystem or download it (respectively).

For example, suppose you are working on a project, _my-project/_, and you want
to depend on targets from your coworker's project, _coworkers-project/_. Both
projects use Bazel, so you can add your coworker's project as an external
dependency and then use any targets your coworker has defined from your own
BUILD files. You would add the following to _my\_project/WORKSPACE_:

```python
local_repository(
    name = "coworkers-project",
    path = "/path/to/coworkers-project",
)
```

If your coworker has a target `//foo:bar`, your project can refer to it as
`@coworkers-project//foo:bar`.

## Depending on non-Bazel projects

Rules prefixed with `new_` (e.g.,
[`new_local_repository`](http://bazel.io/docs/build-encyclopedia.html#new_local_repository)
and [`new_http_archive`](http://bazel.io/docs/build-encyclopedia.html#new_http_archive)
allow you to create targets from projects that do not use Bazel.

For example, suppose you are working on a project, _my-project/_, and you want
to depend on your coworker's project, _coworkers-project/_. Your coworker's
project uses `make` to build, but you'd like to depend on one of the .so files
it generates. To do so, add the following to _my\_project/WORKSPACE_:

```python
new_local_repository(
    name = "coworkers-project",
    path = "/path/to/coworkers-project",
    build_file = "coworker.BUILD",
)
```

`build_file` specifies a BUILD file to overlay on the existing project, for
example:

```python
java_library(
    name = "some-lib",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
```

You can then depend on `@coworkers-project//:some-lib` from your project's BUILD
files.

# Caching of external dependencies

Bazel caches external dependencies and only re-downloads or updates them when
the _WORKSPACE_ file changes. If the _WORKSPACE_ file does not change, Bazel
assumes that the external dependencies have not changed, either. This can cause
unexpected results, especially with local repositories.

For instance, in the example above, suppose that _my-project/_ has a target that
depends on `@coworkers-project//:a`, which you build. Then you change to
_coworkers-project/_ and pull the latest updates to their library, which changes
the behavior of `@coworkers-project//:a`. If you go back to _my-project/_ and
build your target again, it will assume `@coworkers-project//:a` is already
up-to-date and reuse the cached library (instead of realizing that the sources
have changed and, thus, rebuilding).

To avoid this situation, prefer remote repositories to local ones and do not
manually change the files in _\[output\_base\]/external_. If you change a file
in _\[output\_base\]/external_, rerun `bazel fetch ...` to update the cache.

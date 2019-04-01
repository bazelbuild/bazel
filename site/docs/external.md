---
layout: documentation
title: External Dependencies
---

# Working with external dependencies

Bazel can depend on targets from other projects.  Dependencies from these other
projects are called _external dependencies_.

The `WORKSPACE` file in the [workspace directory](build-ref.html#workspace)
tells Bazel how to get other projects' sources.  These other projects can
contain one or more `BUILD` files with their own targets.  `BUILD` files within
the main project can depend on these external targets by using their name from
the `WORKSPACE` file.

For example, suppose there are two projects on a system:

```
/
  home/
    user/
      project1/
        WORKSPACE
        BUILD
        srcs/
          ...
      project2/
        WORKSPACE
        BUILD
        my-libs/
```

If `project1` wanted to depend on a target, `:foo`, defined in
`/home/user/project2/BUILD`, it could specify that a repository named
`project2` could be found at `/home/user/project2`. Then targets in
`/home/user/project1/BUILD` could depend on `@project2//:foo`.

The `WORKSPACE` file allows users to depend on targets from other parts of the
filesystem or downloaded from the internet. Users can also write custom
[repository rules](skylark/repository_rules.html) to get more complex behavior.

This `WORKSPACE` file uses the same syntax as BUILD files, but allows a
different set of rules. The full list of built-in rules are in the Build
Encyclopedia's [Workspace Rules](be/workspace.html) and the documentation
for [Embedded Starklark Repository Rules](repo/index.html).

<a name="types"></a>
## Supported types of external dependencies

A few basic types of external dependencies can be used:

- [Dependencies on other Bazel projects](#bazel-projects)
- [Dependencies on non-Bazel projects](#non-bazel-projects)
- [Dependencies on external packages](#external-packages)

<a name="bazel-projects"></a>
### Depending on other Bazel projects

If you want to use targets from a second Bazel project, you can
use
[`local_repository`](http://docs.bazel.build/be/workspace.html#local_repository),
[`git_repository`](repo/git.html#git_repository)
or [`http_archive`](repo/http.html#http_archive)
to symlink it from the local filesystem, reference a git repository or download
it (respectively).

For example, suppose you are working on a project, `my-project/`, and you want
to depend on targets from your coworker's project, `coworkers-project/`. Both
projects use Bazel, so you can add your coworker's project as an external
dependency and then use any targets your coworker has defined from your own
BUILD files. You would add the following to `my_project/WORKSPACE`:

```python
local_repository(
    name = "coworkers_project",
    path = "/path/to/coworkers-project",
)
```

If your coworker has a target `//foo:bar`, your project can refer to it as
`@coworkers_project//foo:bar`. External project names must be
[valid workspace names](be/functions.html#workspace), so `_` (valid) is used to
replace `-` (invalid) in the name `coworkers_project`.

<a name="non-bazel-projects"></a>
### Depending on non-Bazel projects

Rules prefixed with `new_`, e.g.,
[`new_local_repository`](http://docs.bazel.build/be/workspace.html#new_local_repository),
allow you to create targets from projects that do not use Bazel.

For example, suppose you are working on a project, `my-project/`, and you want
to depend on your coworker's project, `coworkers-project/`. Your coworker's
project uses `make` to build, but you'd like to depend on one of the .so files
it generates. To do so, add the following to `my_project/WORKSPACE`:

```python
new_local_repository(
    name = "coworkers_project",
    path = "/path/to/coworkers-project",
    build_file = "coworker.BUILD",
)
```

`build_file` specifies a BUILD file to overlay on the existing project, for
example:

```python
cc_library(
    name = "some-lib",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
```

You can then depend on `@coworkers_project//:some-lib` from your project's BUILD
files.

<a name="external-packages"></a>
### Depending on external packages

<a name="maven-repositories"></a>
#### Maven repositories

Use the rule [`maven_jar`](https://docs.bazel.build/be/workspace.html#maven_jar)
(and optionally the rule [`maven_server`](https://docs.bazel.build/be/workspace.html#maven_server))
to download a jar from a Maven repository and make it available as a Java
dependency.

<a name="fetching-dependencies"></a>
## Fetching dependencies

By default, external dependencies are fetched as needed during `bazel build`. If
you would like to prefetch the dependencies needed for a specific set of targets, use
[`bazel fetch`](https://docs.bazel.build/versions/master/command-line-reference.html#commands).
To unconditionally fetch all external dependencies, use
[`bazel sync`](https://docs.bazel.build/versions/master/command-line-reference.html#commands).

<a name="shadowing-dependencies"></a>
## Shadowing dependencies

Whenever possible, it is recommended to have a single version policy in your
project. This is required for dependencies that you compile against and end up
in your final binary. But for cases where this isn't true, it is possible to
shadow dependencies. Consider the following scenario:

myproject/WORKSPACE

```python
workspace(name = "myproject")

local_repository(
    name = "A",
    path = "../A",
)
local_repository(
    name = "B",
    path = "../B",
)
```

A/WORKSPACE

```python
workspace(name = "A")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "testrunner",
    urls = ["https://github.com/testrunner/v1.zip"],
    sha256 = "...",
)
```

B/WORKSPACE

```python
workspace(name = "B")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "testrunner",
    urls = ["https://github.com/testrunner/v2.zip"],
    sha256 = "..."
)
```

Both dependencies `A` and `B` depend on `testrunner`, but they depend on
different versions of `testrunner`. There is no reason for these test runners to
not peacefully coexist within `myproject`, however they will clash with each
other since they have the same name. To declare both dependencies,
update myproject/WORKSPACE:

```python
workspace(name = "myproject")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "testrunner-v1",
    urls = ["https://github.com/testrunner/v1.zip"],
    sha256 = "..."
)
http_archive(
    name = "testrunner-v2",
    urls = ["https://github.com/testrunner/v2.zip"],
    sha256 = "..."
)
local_repository(
    name = "A",
    path = "../A",
    repo_mapping = {"@testrunner" : "@testrunner-v1"}
)
local_repository(
    name = "B",
    path = "../B",
    repo_mapping = {"@testrunner" : "@testrunner-v2"}
)
```

This mechanism can also be used to join diamonds. For example if `A` and `B`
had the same dependency but call it by different names, those dependencies can
be joined in myproject/WORKSPACE.


<a name="using-proxies"></a>
## Using Proxies

Bazel will pick up proxy addresses from the `HTTPS_PROXY` and `HTTP_PROXY`
environment variables and use these to download HTTP/HTTPS files (if specified).

<a name="transitive-dependencies"></a>
## Transitive dependencies

Bazel only reads dependencies listed in your `WORKSPACE` file. If your project
(`A`) depends on another project (`B`) which list a dependency on a third
project (`C`) in its `WORKSPACE` file, you'll have to add both `B`
and `C` to your project's `WORKSPACE` file. This requirement can balloon the
`WORKSPACE` file size, but hopefully limits the chances of having one library
include `C` at version 1.0 and another include `C` at 2.0.

Large `WORKSPACE` files can be generated using the tool `generate_workspace`.
For details, see
[Generate external dependencies from Maven projects](generate-workspace.md).

<a name="caching"></a>
## Caching of external dependencies

Bazel caches external dependencies and re-downloads or updates them when
the `WORKSPACE` file changes.

<a name="layout"></a>
## Layout

External dependencies are all downloaded and symlinked under a directory named
`external`. You can see this directory by running:

```
ls $(bazel info output_base)/external
```

Note that running `bazel clean` will not actually delete the external
directory. To remove all external artifacts, use `bazel clean --expunge`.


## Best practices

### Repository rules

Prefer [`http_archive`](repo/http.html#http_archive)
to `git_repository`, `new_git_repository`, and `maven_jar`.
`maven_jar` uses Maven's
internal API, which generally works but is less optimized for Bazel than `http_archive`'s
downloader logic. Track the following issues filed to remediate these problems:

-  [Improve `maven_jar`'s backend.](https://github.com/bazelbuild/bazel/issues/1752)

Do not use `bind()`.  See "[Consider removing
bind](https://github.com/bazelbuild/bazel/issues/1952)" for a long discussion of its issues and
alternatives.

### Custom BUILD files

When using a `new_` repository rule, prefer to specify `build_file_content`, not `build_file`.

### Repository rules

A repository rule should generally be responsible for:

-  Detecting system settings and writing them to files.
-  Finding resources elsewhere on the system.
-  Downloading resources from URLs.
-  Generating or symlinking BUILD files into the external repository directory.

Avoid using `repository_ctx.execute` when possible.  For example, when using a non-Bazel C++
library that has a build using Make, it is preferable to use `repository_ctx.download()` and then
write a BUILD file that builds it, instead of running `ctx.execute(["make"])`.

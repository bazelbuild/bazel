---
layout: documentation
title: External dependencies
---

# Working with external dependencies

Bazel can depend on targets from other projects.  Dependencies from these other
projects are called _external dependencies_.

The `WORKSPACE` file (or `WORKSPACE.bazel` file) in the [workspace directory](build-ref.html#workspace)
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
filesystem or downloaded from the internet. It uses the same syntax as BUILD
files, but allows a different set of rules called _repository rules_ (sometimes
also known as _workspace rules_). Bazel comes with a few [built-in repository
rules](be/workspace.html) and a set of [embedded Starlark repository
rules](repo/index.html). Users can also write [custom repository
rules](skylark/repository_rules.html) to get more complex behavior.

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
[valid workspace names](skylark/lib/globals.html#workspace), so `_` (valid) is used to
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
#### Maven artifacts and repositories

Use the ruleset [`rules_jvm_external`](https://github.com/bazelbuild/rules_jvm_external)
to download artifacts from Maven repositories and make them available as Java
dependencies.

<a name="fetching-dependencies"></a>
## Fetching dependencies

By default, external dependencies are fetched as needed during `bazel build`. If
you would like to prefetch the dependencies needed for a specific set of targets, use
[`bazel fetch`](https://docs.bazel.build/versions/master/command-line-reference.html#commands).
To unconditionally fetch all external dependencies, use
[`bazel sync`](https://docs.bazel.build/versions/master/command-line-reference.html#commands).
As fetched repositories are [stored in the output base](#layout), fetching
happens per workspace.

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

## Overriding repositories from the command line

To override a declared repository with a local repository from the command line,
use the
[`--override_repository`](command-line-reference.html#flag--override_repository)
flag. Using this flag changes the contents of external repositories without
changing your source code.

For example, to override `@foo` to the local directory `/path/to/local/foo`,
pass the `--override_repository=foo=/path/to/local/foo` flag.

Some of the use cases include:

* Debugging issues. For example, you can override a `http_archive` repository
  to a local directory where you can make changes more easily.
* Vendoring. If you are in an environment where you cannot make network calls,
  override the network-based repository rules to point to local directories
  instead.

<a name="using-proxies"></a>
## Using proxies

Bazel will pick up proxy addresses from the `HTTPS_PROXY` and `HTTP_PROXY`
environment variables and use these to download HTTP/HTTPS files (if specified).

<a name="transitive-dependencies"></a>
## Transitive dependencies

Bazel only reads dependencies listed in your `WORKSPACE` file. If your project
(`A`) depends on another project (`B`) which lists a dependency on a third
project (`C`) in its `WORKSPACE` file, you'll have to add both `B`
and `C` to your project's `WORKSPACE` file. This requirement can balloon the
`WORKSPACE` file size, but limits the chances of having one library
include `C` at version 1.0 and another include `C` at 2.0.

<a name="caching"></a>
## Caching of external dependencies

By default, Bazel will only re-download external dependencies if their
definition changes. Changes to files referenced in the definition (e.g., patches
or `BUILD` files) are also taken into account by bazel.

To force a re-download, use `bazel sync`.


<a name="layout"></a>
## Layout

External dependencies are all downloaded to a directory under the subdirectory
`external` in the [output base](output_directories.html). In case of a
[local repository](be/workspace.html#local_repository), a symlink is created
there instead of creating a new directory.
You can see the `external` directory by running:

```
ls $(bazel info output_base)/external
```

Note that running `bazel clean` will not actually delete the external
directory. To remove all external artifacts, use `bazel clean --expunge`.

## Offline builds

It is sometimes desirable or necessary to run a build in an offline fashion. For
simple use cases, e.g., traveling on an airplane,
[prefetching](#fetching-dependencies) the needed
repositories with `bazel fetch` or `bazel sync` can be enough; moreover, the
using the option `--nofetch`, fetching of further repositories can be disabled
during the build.

For true offline builds, where the providing of the needed files is to be done
by an entity different from bazel, bazel supports the option
`--distdir`. Whenever a repository rule asks bazel to fetch a file via
[`ctx.download`](skylark/lib/repository_ctx.html#download) or
[`ctx.download_and_extract`](skylark/lib/repository_ctx.html#download_and_extract)
and provides a hash sum of the file
needed, bazel will first look into the directories specified by that option for
a file matching the basename of the first URL provided, and use that local copy
if the hash matches.

Bazel itself uses this technique to bootstrap offline from the [distribution
artifact](https://bazel.build/designs/2016/10/11/distribution-artifact.html).
It does so by [collecting all the needed external
dependencies](https://github.com/bazelbuild/bazel/blob/5cfa0303d6ac3b5bd031ff60272ce80a704af8c2/WORKSPACE#L116)
in an internal
[`distdir_tar`](https://github.com/bazelbuild/bazel/blob/5cfa0303d6ac3b5bd031ff60272ce80a704af8c2/distdir.bzl#L44).

However, bazel allows the execution of arbitrary commands in repository rules,
without knowing if they call out to the network. Therefore, bazel has no option
to enforce builds being fully offline. So testing if a build works correctly
offline requires external blocking of the network, as bazel does in its
bootstrap test.

## Best practices

### Repository rules

Prefer [`http_archive`](repo/http.html#http_archive) to `git_repository` and
`new_git_repository`. The reasons are:

* Git repository rules depend on system `git(1)` whereas the HTTP downloader is built
  into Bazel and has no system dependencies.
* `http_archive` supports a list of `urls` as mirrors, and `git_repository` supports only
  a single `remote`.
* `http_archive` works with the [repository cache](guide.html#repository-cache), but not
  `git_repository`. See
   [#5116](https://github.com/bazelbuild/bazel/issues/5116) for more information.


Do not use `bind()`.  See "[Consider removing
bind](https://github.com/bazelbuild/bazel/issues/1952)" for a long discussion of its issues and
alternatives.

### Repository rules

A repository rule should generally be responsible for:

-  Detecting system settings and writing them to files.
-  Finding resources elsewhere on the system.
-  Downloading resources from URLs.
-  Generating or symlinking BUILD files into the external repository directory.

Avoid using `repository_ctx.execute` when possible.  For example, when using a non-Bazel C++
library that has a build using Make, it is preferable to use `repository_ctx.download()` and then
write a BUILD file that builds it, instead of running `ctx.execute(["make"])`.

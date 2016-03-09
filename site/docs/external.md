---
layout: documentation
title: External Dependencies
---

# Working with external dependencies

Bazel is designed to have absolutely everything needed for a build, from source
code to libraries to compilers, under one directory (the workspace directory).
This is impractical for some version control systems and goes against how many
existing projects are structured. Thus, Bazel has a system for pulling in
dependencies from outside of the workspace.

External dependencies can be specified in a `WORKSPACE` file in the
[workspace directory](/docs/build-ref.html#workspaces). This `WORKSPACE` file
uses the same Python-like syntax of BUILD files, but allows a different set of
rules. See the full list of rules that are allowed in the
[Workspace](/docs/be/workspace.html) list of rules in the Build
Encyclopedia.

External dependencies are all downloaded and symlinked under a directory named
`external`. You can see this directory by running:

```
ls $(bazel info output_base)/external
```

Note that running `bazel clean` will not actually delete the external
directory: to remove all external artifacts, use `bazel clean --expunge`.

## Fetching dependencies

By default, external dependencies are fetched as needed during `bazel build`. If
you would like to disable this behavior or prefetch dependencies, use
[`bazel fetch`](http://bazel.io/docs/bazel-user-manual.html#fetch).

## Using Proxies

Bazel will pick up proxy addresses from the `HTTPS_PROXY` and `HTTP_PROXY`
environment variables and use these to download HTTP/HTTPS files (if specified).

<a name="transitive-dependencies"></a>
## Transitive dependencies

Bazel only reads dependencies listed in your `WORKSPACE` file. This
means that if your project (`A`) depends on another project (`B`) which list a
dependency on project `C` in its `WORKSPACE` file, you'll have to add both `B`
and `C` to your project's `WORKSPACE` file. This can balloon the `WORKSPACE`
file size, but hopefully limits the chances of having one library include `C`
at version 1.0 and another include `C` at 2.0.

Bazel provides a tool to help generate these expansive `WORKSPACE` files, called
`generate_workspace`. This is not included with the binary installer, so you'll
need to clone the [GitHub repo](https://github.com/bazelbuild/bazel) to use it.
We recommend using the tag corresponding to your current version of bazel, which
you can check by running `bazel version`.

`cd` to the GitHub clone, `git checkout` the appropriate tag, and run the
following to build the tool and see usage:

```
bazel run //src/tools/generate_workspace
```

You can specify directories containing Bazel projects (i.e., directories
containing a `WORKSPACE` file), Maven projects (i.e., directories containing a
`pom.xml` file), or Maven artifact coordinates directly. For example:

```bash
$ bazel run //src/tools/generate_workspace -- \
>    --maven_project=/path/to/my/project \
>    --bazel_project=/path/to/skunkworks \
>    --bazel_project=/path/to/teleporter/project \
>    --artifact=groupId:artifactId:version \
>    --artifact=groupId:artifactId:version
Wrote:
/tmp/1437415510621-0/2015-07-20-14-05-10.WORKSPACE
/tmp/1437415510621-0/2015-07-20-14-05-10.BUILD
```

The `WORKSPACE` file will contain the transitive dependencies of the given
projects and artifacts. The `BUILD` file will contain a single target,
`transitive-deps`, that contains all of the dependencies. You can copy these
files to your project and add `transitive-deps` as a dependency of your `java_`
targets in `BUILD` files.

If you specify multiple Bazel projects, Maven projects, or artifacts, they will
all be combined into one `WORKSPACE` file (e.g., if the Bazel project depends on
junit and the Maven project also depends on junit, junit will only appear once
as a dependency in the output).

You may wish to curate the generated `WORKSPACE` file to ensure it is using the
correct version of each dependency. If several different versions of an artifact
are requested (by different libraries that depend on it), then
`generate_workspace` chooses a version and annotates the `maven_jar` with the
other versions requested, for example:

```python
# org.springframework:spring:2.5.6
# javax.mail:mail:1.4
# httpunit:httpunit:1.6 wanted version 1.0.2
# org.springframework:spring-support:2.0.2 wanted version 1.0.2
# org.slf4j:nlog4j:1.2.24 wanted version 1.0.2
maven_jar(
    name = "javax/activation/activation",
    artifact = "javax.activation:activation:1.1",
)
```

This indicates that `org.springframework:spring:2.5.6`, `javax.mail:mail:1.4`,
`httpunit:httpunit:1.6`, `org.springframework:spring-support:2.0.2`, and
`org.slf4j:nlog4j:1.2.24` all depend on javax.activation. However, two of these
libraries wanted version 1.1 and three of them wanted 1.0.2. The `WORkSPACE`
file is using version 1.1, but that might not be the right version to use.

You may also want to break `transitive-deps` into smaller targets, as it is
unlikely that all of your targets depend on the transitive closure of your
maven jars.

# Types of external dependencies

There are a few basic types of external dependencies that can be created.

## Combining Bazel projects

If you have a second Bazel project that you'd like to use targets from, you can
use
[`local_repository`](http://bazel.io/docs/be/workspace.html#local_repository)
or [`http_archive`](http://bazel.io/docs/be/workspace.html#http_archive)
to symlink it from the local filesystem or download it (respectively).

For example, suppose you are working on a project, `my-project/`, and you want
to depend on targets from your coworker's project, `coworkers-project/`. Both
projects use Bazel, so you can add your coworker's project as an external
dependency and then use any targets your coworker has defined from your own
BUILD files. You would add the following to `my_project/WORKSPACE`:

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
[`new_local_repository`](http://bazel.io/docs/be/workspace.html#new_local_repository)
and [`new_http_archive`](http://bazel.io/docs/be/workspace.html#new_http_archive)
) allow you to create targets from projects that do not use Bazel.

For example, suppose you are working on a project, `my-project/`, and you want
to depend on your coworker's project, `coworkers-project/`. Your coworker's
project uses `make` to build, but you'd like to depend on one of the .so files
it generates. To do so, add the following to `my_project/WORKSPACE`:

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
the `WORKSPACE` file changes. If the `WORKSPACE` file does not change, Bazel
assumes that the external dependencies have not changed, either. This can cause
unexpected results, especially with local repositories.

For instance, in the example above, suppose that `my-project/` has a target that
depends on `@coworkers-project//:a`, which you build. Then you change to
`coworkers-project/` and pull the latest updates to their library, which changes
the behavior of `@coworkers-project//:a`. If you go back to `my-project/` and
build your target again, it will assume `@coworkers-project//:a` is already
up-to-date and reuse the cached library (instead of realizing that the sources
have changed and, thus, rebuilding).

To avoid this situation, prefer remote repositories to local ones and do not
manually change the files in `[output_base]/external`. If you change a file
in `[output_base]/external`, rerun `bazel fetch ...` to update the cache.

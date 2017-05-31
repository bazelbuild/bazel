---
layout: documentation
title: External Dependencies
---

# Working with external dependencies

External dependencies can be specified in the `WORKSPACE` file of the
[workspace directory](/docs/build-ref.html#workspace). This `WORKSPACE` file
uses the same syntax as BUILD files, but allows a different set of
rules. The full list of rules are in the Build Encyclopedia's
[Workspace Rules](/docs/be/workspace.html).

External dependencies are all downloaded and symlinked under a directory named
`external`. You can see this directory by running:

```
ls $(bazel info output_base)/external
```

Note that running `bazel clean` will not actually delete the external
directory. To remove all external artifacts, use `bazel clean --expunge`.

## Supported types of external dependencies

A few basic types of external dependencies can be used:

- [Dependencies on other Bazel projects](#bazel-projects)
- [Dependencies on non-Bazel projects](#non-bazel-projects)
- [Dependencies on external packages](#external-packages)

<a name="bazel-projects"></a>
### Depending on other Bazel projects

If you want to use targets from a second Bazel project, you can
use
[`local_repository`](http://bazel.build/docs/be/workspace.html#local_repository),
[`git_repository`](https://bazel.build/docs/be/workspace.html#git_repository)
or [`http_archive`](http://bazel.build/docs/be/workspace.html#http_archive)
to symlink it from the local filesystem, reference a git repository or download
it (respectively).

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

<a name="non-bazel-projects"></a>
### Depending on non-Bazel projects

Rules prefixed with `new_` (e.g.,
[`new_local_repository`](http://bazel.build/docs/be/workspace.html#new_local_repository),
[`new_git_repository`](https://bazel.build/docs/be/workspace.html#new_git_repository)
and [`new_http_archive`](http://bazel.build/docs/be/workspace.html#new_http_archive)
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

<a name="external-packages"></a>
### Depending on external packages

#### Maven repositories

Use the rule [`maven_jar`](https://bazel.build/versions/master/docs/be/workspace.html#maven_jar)
(and optionally the rule [`maven_server`](https://bazel.build/versions/master/docs/be/workspace.html#maven_server))
to download a jar from a Maven repository and make it available as a Java
dependency.

## Fetching dependencies

By default, external dependencies are fetched as needed during `bazel build`. If
you would like to disable this behavior or prefetch dependencies, use
[`bazel fetch`](http://bazel.build/docs/bazel-user-manual.html#fetch).

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

## Generate a `WORKSPACE` file

Bazel provides a tool to help generate these expansive `WORKSPACE` files, called
`generate_workspace`. This tool is not included with the binary installer, so
you'll need to clone the [GitHub repo](https://github.com/bazelbuild/bazel) to
use it. We recommend using the tag corresponding to your current version of
bazel, which you can check by running `bazel version`.

`cd` to the GitHub clone, `git checkout` the appropriate tag, and run the
following to build the tool and see usage:

```
bazel run //src/tools/generate_workspace
```

Note that you need run this command from your Bazel source folder even if you
build your binary from source.

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
    name = "javax_activation_activation",
    artifact = "javax.activation:activation:1.1",
)
```

The example above indicates that `org.springframework:spring:2.5.6`,
`javax.mail:mail:1.4`, `httpunit:httpunit:1.6`,
`org.springframework:spring-support:2.0.2`, and `org.slf4j:nlog4j:1.2.24`
all depend on javax.activation. However, two of these libraries wanted
version 1.1 and three of them wanted 1.0.2. The `WORKSPACE` file is using
version 1.1, but that might not be the right version to use.

You may also want to break `transitive-deps` into smaller targets, as it is
unlikely that all of your targets depend on the transitive closure of your
maven jars.

## Caching of external dependencies

Bazel caches external dependencies and re-downloads or updates them when
the `WORKSPACE` file changes.

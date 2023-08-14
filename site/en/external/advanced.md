Project: /_project.yaml
Book: /_book.yaml

# Advanced topics on external dependencies

{% include "_buttons.html" %}

## Shadowing dependencies in WORKSPACE

Note: This section applies to the [WORKSPACE
system](/external/overview#workspace-system) only. For
[Bzlmod](/external/overview#bzlmod), use a [multiple-version
override](/external/module#multiple-version_override).

Whenever possible, have a single version policy in your project, which is
required for dependencies that you compile against and end up in your final
binary. For other cases, you can shadow dependencies:

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

B/WORKSPACE {# This is not a buganizer link okay?? #}

```python
workspace(name = "B")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "testrunner",
    urls = ["https://github.com/testrunner/v2.zip"],
    sha256 = "..."
)
```

Both dependencies `A` and `B` depend on different versions of `testrunner`.
Include both in `myproject` without conflict by giving them distinct names in
`myproject/WORKSPACE`:

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

You can also use this mechanism to join diamonds. For example, if `A` and `B`
have the same dependency but call it by different names, join those dependencies
in `myproject/WORKSPACE`.

## Overriding repositories from the command line {:#overriding-repositories}

To override a declared repository with a local repository from the command line,
use the
[`--override_repository`](/reference/command-line-reference#flag--override_repository)
flag. Using this flag changes the contents of external repositories without
changing your source code.

For example, to override `@foo` to the local directory `/path/to/local/foo`,
pass the `--override_repository=foo=/path/to/local/foo` flag.

Use cases include:

*   Debugging issues. For example, to override an `http_archive` repository to a
    local directory where you can make changes more easily.
*   Vendoring. If you are in an environment where you cannot make network calls,
    override the network-based repository rules to point to local directories
    instead.

Note: With [Bzlmod](/external/overview#bzlmod), remember to use canonical repo
names here. Alternatively, use the
[`--override_module`](/reference/command-line-reference#flag--override_module)
flag to override a module to a local directory, similar to the
[`local_path_override`](/rules/lib/globals/module#local_path_override) directive in
`MODULE.bazel`.

## Using proxies

Bazel picks up proxy addresses from the `HTTPS_PROXY` and `HTTP_PROXY`
environment variables and uses these to download `HTTP` and `HTTPS` files (if
specified).

## Support for IPv6

On IPv6-only machines, Bazel can download dependencies with no changes. However,
on dual-stack IPv4/IPv6 machines Bazel follows the same convention as Java,
preferring IPv4 if enabled. In some situations, for example when the IPv4
network cannot resolve/reach external addresses, this can cause `Network
unreachable` exceptions and build failures. In these cases, you can override
Bazel's behavior to prefer IPv6 by using the
[`java.net.preferIPv6Addresses=true` system
property](https://docs.oracle.com/javase/8/docs/api/java/net/doc-files/net-properties.html){: .external}.
Specifically:

*   Use `--host_jvm_args=-Djava.net.preferIPv6Addresses=true` [startup
    option](/docs/user-manual#startup-options), for example by adding the
    following line in your [`.bazelrc` file](/run/bazelrc):

    `startup --host_jvm_args=-Djava.net.preferIPv6Addresses=true`

*   When running Java build targets that need to connect to the internet (such
    as for integration tests), use the
    `--jvmopt=-Djava.net.preferIPv6Addresses=true` [tool
    flag](/docs/user-manual#jvmopt). For example, include in your [`.bazelrc`
    file](/run/bazelrc):

    `build --jvmopt=-Djava.net.preferIPv6Addresses`

*   If you are using [`rules_jvm_external`](https://github.com/bazelbuild/rules_jvm_external){: .external}
    for dependency version resolution, also add
    `-Djava.net.preferIPv6Addresses=true` to the `COURSIER_OPTS` environment
    variable to [provide JVM options for
    Coursier](https://github.com/bazelbuild/rules_jvm_external#provide-jvm-options-for-coursier-with-coursier_opts){: .external}.

## Offline builds

Sometimes you may wish to run a build offline, such as when traveling on an
airplane. For such simple use cases, prefetch the needed repositories with
`bazel fetch` or `bazel sync`. To disable fetching further repositories during
the build, use the option `--nofetch`.

For true offline builds, where a different entity supplies all needed files,
Bazel supports the option `--distdir`. This flag tells Bazel to look first into
the directories specified by that option when a repository rule asks Bazel to
fetch a file with [`ctx.download`](/rules/lib/builtins/repository_ctx#download) or
[`ctx.download_and_extract`](/rules/lib/builtins/repository_ctx#download_and_extract). By
providing a hash sum of the file needed, Bazel looks for a file matching the
basename of the first URL, and uses the local copy if the hash matches.

Bazel itself uses this technique to bootstrap offline from the [distribution
artifact](https://github.com/bazelbuild/bazel-website/blob/master/designs/_posts/2016-10-11-distribution-artifact.md).
It does so by [collecting all the needed external
dependencies](https://github.com/bazelbuild/bazel/blob/5cfa0303d6ac3b5bd031ff60272ce80a704af8c2/WORKSPACE#L116){: .external}
in an internal
[`distdir_tar`](https://github.com/bazelbuild/bazel/blob/5cfa0303d6ac3b5bd031ff60272ce80a704af8c2/distdir.bzl#L44){: .external}.

Bazel allows execution of arbitrary commands in repository rules without knowing
if they call out to the network, and so cannot enforce fully offline builds. To
test if a build works correctly offline, manually block off the network (as
Bazel does in its [bootstrap
test](https://cs.opensource.google/bazel/bazel/+/master:src/test/shell/bazel/BUILD;l=1073;drc=88c426e73cc0eb0a41c0d7995e36acd94e7c9a48){: .external}).
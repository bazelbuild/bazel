Project: /_project.yaml
Book: /_book.yaml


# Optimize Iteration Speed

{% include "_buttons.html" %}

This page describes how to optimize Bazel's build performance when running Bazel
repeatedly.

## Bazel's Runtime State

A Bazel invocation involves several interacting parts.

*   The `bazel` command line interface (CLI) is the user-facing front-end tool
    and receives commands from the user.

*   The CLI tool starts a [*Bazel server*](https://bazel.build/run/client-server)
    for each distinct [output base](https://bazel.build/remote/output-directories).
    The Bazel server is generally persistent, but will shut down after some idle
    time so as to not waste resources.

*   The Bazel server performs the loading and analysis steps for a given command
    (`build`, `run`, `cquery`, etc.), in which it constructs the necessary parts
    of the build graph in memory. The resulting data structures are retained in
    the Bazel server as part of the *analysis cache*.

*   The Bazel server can also perform the action execution, or it can send
    actions off for remote execution if it is set up to do so. The results of
    action executions are also cached, namely in the *action cache* (or
    *execution cache*, which may be either local or remote, and it may be shared
    among Bazel servers).

*   The result of the Bazel invocation is made available in the output tree.

## Running Bazel Iteratively

In a typical developer workflow, it is common to build (or run) a piece of code
repeatedly, often at a very high frequency (e.g. to resolve some compilation
error or investigate a failing test). In this situation, it is important that
repeated invocations of `bazel` have as little overhead as possible relative to
the underlying, repeated action (e.g. invoking a compiler, or executing a test).

With this in mind, we take another look at Bazel's runtime state:

The analysis cache is a critical piece of data. A significant amount of time can
be spent just on the loading and analysis phases of a cold run (i.e. a run just
after the Bazel server was started or when the analysis cache was discarded).
For a single, successful cold build (e.g. for a production release) this cost is
bearable, but for repeatedly building the same target it is important that this
cost be amortized and not repeated on each invocation.

The analysis cache is rather volatile. First off, it is part of the in-process
state of the Bazel server, so losing the server loses the cache. But the cache
is also *invalidated* very easily: for example, many `bazel` command line flags
cause the cache to be discarded. This is because many flags affect the build
graph (e.g. because of
[configurable attributes](https://bazel.build/configure/attributes)). Some flag
changes can also cause the Bazel server to be restarted (e.g. changing
[startup options](https://bazel.build/docs/user-manual#startup-options)).

Bazel will print a warning if either the analysis cache was discarded or the
server was restarted. Either of these should be avoided during iterative use:

*   Be mindful of changing `bazel` flags in the middle of an iterative
    workflow. For example, mixing a `bazel build -c opt` with a `bazel cquery`
    causes each command to discard the analysis cache of the other. In general,
    try to use a fixed set of flags for the duration of a particular workflow.

*   Losing the Bazel server loses the analysis cache. The Bazel server has a
    [configurable](https://bazel.build/docs/user-manual#max-idle-secs) idle
    time, after which it shuts down. You can configure this time via your
    bazelrc file to suit your needs. The server also restarted when startup
    flags change, so, again, avoid changing those flags if possible.

*   If you want to use multiple sets of flags from the same workspace, you can
    use multiple, distinct output bases, switched with the `--output_base`
    flag. Each output base gets its own Bazel server.

A good execution cache is also valuable for build performance. An execution
cache can be kept locally
[on disk](https://bazel.build/remote/caching#disk-cache), or
[remotely](https://bazel.build/remote/caching). The cache can be shared among
Bazel servers, and indeed among developers.

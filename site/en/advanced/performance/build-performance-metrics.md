Project: /_project.yaml
Book: /_book.yaml

# Extracting build performance metrics

{% include "_buttons.html" %}

Probably every Bazel user has experienced builds that were slow or slower than
anticipated. Improving the performance of individual builds has particular value
for targets with significant impact, such as:

1. Core developer targets that are frequently iterated on and (re)built.

2. Common libraries widely depended upon by other targets.

3. A representative target from a class of targets (e.g. custom rules),
  diagnosing and fixing issues in one build might help to resolve issues at the
  larger scale.

An important step to improving the performance of builds is to understand where
resources are spent. This page lists different metrics you can collect.
[Breaking down build performance](/configure/build-performance-breakdown) showcases
how you can use these metrics to detect and fix build performance issues.

There are a few main ways to extract metrics from your Bazel builds, namely:

## Build Event Protocol (BEP) {:#build-event-protocol}

Bazel outputs a variety of protocol buffers
[`build_event_stream.proto`](https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/buildeventstream/proto/build_event_stream.proto)
through the [Build Event Protocol (BEP)](/remote/bep), which
can be aggregated by a backend specified by you. Depending on your use cases,
you might decide to aggregate the metrics in various ways, but here we will go
over some concepts and proto fields that would be useful in general to consider.

## Bazel’s query / cquery / aquery commands {:#bazel-commands-query-cquery-aquery}

Bazel provides 3 different query modes ([query](/query/quickstart),
[cquery](/query/cquery) and [aquery](/query/aquery)) that allow users
to query the target graph, configured target graph and action graph
respectively. The query language provides a
[suite of functions](/query/language#functions) usable across the different
query modes, that allows you to customize your queries according to your needs.

## JSON Trace Profiles {:#json-trace-profiles}

For every build-like Bazel invocation, Bazel writes a trace profile in JSON
format. The [JSON trace profile](/advanced/performance/json-trace-profile) can
be very useful to quickly understand what Bazel spent time on during the
invocation.

## Execution Log {:#execution-log}

The [execution log](/remote/cache-remote) can help you to troubleshoot and fix
missing remote cache hits due to machine and environment differences or
non-deterministic actions. If you pass the flag
[`--experimental_execution_log_spawn_metrics`](/reference/command-line-reference#flag--experimental_execution_log_spawn_metrics)
(available from Bazel 5.2) it will also contain detailed spawn metrics, both for
locally and remotely executed actions. You can use these metrics for example to
make comparisons between local and remote machine performance or to find out
which part of the spawn execution is consistently slower than expected (for
example due to queuing).

## Execution Graph Log {:#execution-graph-log}

While the JSON trace profile contains the critical path information, sometimes
you need additional information on the dependency graph of the executed actions.
Starting with Bazel 6.0, you can pass the flags
`--experimental_execution_graph_log` and
`--experimental_execution_graph_log_dep_type=all` to write out a log about the
executed actions and their inter-dependencies.

This information can be used to understand the drag that is added by a node on
the critical path. The drag is the amount of time that can potentially be saved
by removing a particular node from the execution graph.

The data helps you predict the impact of changes to the build and action graph
before you actually do them.

## Benchmarking with bazel-bench {:#bazel-bench}

[Bazel bench](https://github.com/bazelbuild/bazel-bench) is a
benchmarking tool for Git projects to benchmark build performance in the
following cases:

* **Project benchmark:** Benchmarking two git commits against each other at a
 single Bazel version. Used to detect regressions in your build (often through
 the addition of dependencies).

* **Bazel benchmark:** Benchmarking two versions of Bazel against each other at
 a single git commit. Used to detect regressions within Bazel itself (if you
 happen to maintain / fork Bazel).

Benchmarks monitor wall time, CPU  time and system time and Bazel’s retained
heap size.

It is also recommended to run Bazel bench on dedicated, physical machines that
are not running other processes so as to reduce sources of variability.

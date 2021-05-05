---
layout: documentation
title: Build Event Protocol Glossary
---

# Build Event Protocol Glossary


Each BEP event type has its own semantics, minimally documented in
[build\_event\_stream.proto](https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/buildeventstream/proto/build_event_stream.proto).
The following glossary describes each event type.

## Aborted

Unlike other events, `Aborted` does not have a corresponding ID type, because
the `Aborted` event *replaces* events of other types. This event indicates that
the build terminated early and the event ID it appears under was not produced
normally. `Aborted` contains an enum and human-friendly description to explain
why the build did not complete.

For example, if a build is evaluating a target when the user interrupts Bazel,
BEP contains an event like the following:

<p>
  <button class="btn btn-primary" type="button" data-toggle="collapse"
      data-target="#collapseAbortedJson" aria-expanded="false"
      aria-controls="collapseAbortedJson">
    Show/Hide BEP JSON
  </button>
</p>
{: .collapse #collapseAbortedJson}
<p>
  <button class="btn btn-primary" type="button" data-toggle="collapse"
      data-target="#collapseBuildMetricsJson" aria-expanded="false"
      aria-controls="collapseBuildMetricsJson">
    Show/Hide BEP JSON
  </button>
</p>
{: .collapse #collapseBuildMetricsJson}
<p>
  <button class="btn btn-primary" type="button" data-toggle="collapse"
      data-target="#collapseBuildToolLogsJson" aria-expanded="false"
      aria-controls="collapseBuildToolLogsJson">
    Show/Hide BEP JSON
  </button>
</p>
{: .collapse #collapseBuildToolLogsJson}
<p>
  <button class="btn btn-primary" type="button" data-toggle="collapse"
      data-target="#collapseConfigurationJson" aria-expanded="false"
      aria-controls="collapseConfigurationJson">
    Show/Hide BEP JSON
  </button>
</p>
{: .collapse #collapseConfigurationJson}
<p>
  <button class="btn btn-primary" type="button" data-toggle="collapse"
      data-target="#collapseConvenienceSymlinksIdentifiedJson"
      aria-expanded="false"
      aria-controls="collapseConvenienceSymlinksIdentifiedJson">
    Show/Hide BEP JSON
  </button>
</p>
{: .collapse #collapseConvenienceSymlinksIdentifiedJson}
For more information on interpreting a stream's `NamedSetOfFiles` events, see the
[BEP examples page](bep-examples.html#consuming-namedsetoffiles).

## OptionsParsed

A single `OptionsParsed` event lists all options applied to the command,
separating startup options from command options. It also includes the
[InvocationPolicy](https://docs.bazel.build/command-line-reference.html#flag--invocation_policy), if any.

<p>
  <button class="btn btn-primary" type="button" data-toggle="collapse"
      data-target="#collapseOptionsParsedJson" aria-expanded="false"
      aria-controls="collapseOptionsParsedJson">
    Show/Hide BEP JSON
  </button>
</p>
{: .collapse #collapseOptionsParsedJson}
<p>
  <button class="btn btn-primary" type="button" data-toggle="collapse"
      data-target="#collapsePatternExpandedJson" aria-expanded="false"
      aria-controls="collapsePatternExpandedJson">
    Show/Hide BEP JSON
  </button>
</p>
{: .collapse #collapsePatternExpandedJson}
<p>
  <button class="btn btn-primary" type="button" data-toggle="collapse"
      data-target="#collapseTargetCompleteJson" aria-expanded="false"
      aria-controls="collapseTargetCompleteJson">
    Show/Hide BEP JSON
  </button>
</p>
{: .collapse #collapseTargetCompleteJson}
<p>
  <button class="btn btn-primary" type="button" data-toggle="collapse"
      data-target="#collapseTargetConfiguredJson" aria-expanded="false"
      aria-controls="collapseTargetConfiguredJson">
    Show/Hide BEP JSON
  </button>
</p>
{: .collapse #collapseTargetConfiguredJson}

## TargetSummary

For each `(target, configuration)` pair that is executed, a `TargetSummary`
event is included with an aggregate success result encompassing the configured
target's execution and all aspects applied to that configured target.

## TestResult

If testing is requested, a `TestResult` event is sent for each test attempt,
shard, and run per test. This allows BEP consumers to identify precisely which
test actions failed their tests and identify the test outputs (e.g. logs,
test.xml files) for each test action.

## TestSummary

If testing is requested, a `TestSummary` event is sent for each test `(target,
configuration)`, containing information necessary to interpret the test's
results. The number of attempts, shards and runs per test are included to enable
BEP consumers to differentiate artifacts across these dimensions.  The attempts
and runs per test are considered while producing the aggregate `TestStatus` to
differentiate `FLAKY` tests from `FAILED` tests.

## UnstructuredCommandLine

Unlike [CommandLine](#commandline), this event carries the unparsed commandline
flags in string form as encountered by the build tool after expanding all
[`.bazelrc`](guide.html#bazelrc-the-bazel-configuration-file) files and
considering the `--config` flag.

The `UnstructuredCommandLine` event may be relied upon to precisely reproduce a
given command execution.

## WorkspaceConfig

A single `WorkspaceConfig` event contains configuration information regarding the
workspace, such as the execution root.

## WorkspaceStatus

A single `WorkspaceStatus` event contains the result of the [workspace status
command](user-manual.html#workspace_status).

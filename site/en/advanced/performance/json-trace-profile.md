Project: /_project.yaml
Book: /_book.yaml

# JSON Trace Profile

{% include "_buttons.html" %}

The JSON trace profile can be very useful to quickly understand what Bazel spent
time on during the invocation.

By default, for all build-like commands and query Bazel writes such a profile to
`command.profile.gz`. You can configure whether a profile is written with the
[`--generate_json_trace_profile`](/reference/command-line-reference#flag--generate_json_trace_profile)
flag, and the location it is written to with the
[`--profile`](/docs/user-manual#profile) flag. Locations ending with `.gz` are
compressed with GZIP. Use the flag
[`--experimental_announce_profile_path`](/reference/command-line-reference#flag--experimental_announce_profile_path)
to print the path to this file to the log.

## Tools

You can load this profile into `chrome://tracing` or analyze and
post-process it with other tools.

### `chrome://tracing`

To visualize the profile, open `chrome://tracing` in a Chrome browser tab,
click "Load" and pick the (potentially compressed) profile file. For more
detailed results, click the boxes in the lower left corner.

Example profile:

![Example profile](/docs/images/json-trace-profile.png "Example profile")

**Figure 1.** Example profile.

You can use these keyboard controls to navigate:

*   Press `1` for "select" mode. In this mode, you can select
    particular boxes to inspect the event details (see lower left corner).
    Select multiple events to get a summary and aggregated statistics.
*   Press `2` for "pan" mode. Then drag the mouse to move the view. You
    can also use `a`/`d` to move left/right.
*   Press `3` for "zoom" mode. Then drag the mouse to zoom. You can
    also use `w`/`s` to zoom in/out.
*   Press `4` for "timing" mode where you can measure the distance
    between two events.
*   Press `?` to learn about all controls.

### `bazel analyze-profile`

The Bazel subcommand [`analyze-profile`](/docs/user-manual#analyze-profile)
consumes a profile format and prints cumulative statistics for
different task types for each build phase and an analysis of the critical path.

For example, the commands

```
$ bazel build --profile=/tmp/profile.gz //path/to:target
...
$ bazel analyze-profile /tmp/profile.gz
```

may yield output of this form:

```
INFO: Profile created on Tue Jun 16 08:59:40 CEST 2020, build ID: 0589419c-738b-4676-a374-18f7bbc7ac23, output base: /home/johndoe/.cache/bazel/_bazel_johndoe/d8eb7a85967b22409442664d380222c0

=== PHASE SUMMARY INFORMATION ===

Total launch phase time         1.070 s   12.95%
Total init phase time           0.299 s    3.62%
Total loading phase time        0.878 s   10.64%
Total analysis phase time       1.319 s   15.98%
Total preparation phase time    0.047 s    0.57%
Total execution phase time      4.629 s   56.05%
Total finish phase time         0.014 s    0.18%
------------------------------------------------
Total run time                  8.260 s  100.00%

Critical path (4.245 s):
       Time Percentage   Description
    8.85 ms    0.21%   _Ccompiler_Udeps for @local_config_cc// compiler_deps
    3.839 s   90.44%   action 'Compiling external/com_google_protobuf/src/google/protobuf/compiler/php/php_generator.cc [for host]'
     270 ms    6.36%   action 'Linking external/com_google_protobuf/protoc [for host]'
    0.25 ms    0.01%   runfiles for @com_google_protobuf// protoc
     126 ms    2.97%   action 'ProtoCompile external/com_google_protobuf/python/google/protobuf/compiler/plugin_pb2.py'
    0.96 ms    0.02%   runfiles for //tools/aquery_differ aquery_differ
```

### Bazel Invocation Analyzer

The open-source
[Bazel Invocation Analyzer](https://github.com/EngFlow/bazel_invocation_analyzer){: .external}
consumes a profile format and prints suggestions on how to improve
the build’s performance. This analysis can be performed using its CLI or on
[https://analyzer.engflow.com](https://analyzer.engflow.com){: .external}.

### `jq`

`jq` is like `sed` for JSON data. An example usage of `jq` to extract all
durations of the sandbox creation step in local action execution:

```
$ zcat $(../bazel-6.0.0rc1-linux-x86_64 info output_base)/command.profile.gz | jq '.traceEvents | .[] | select(.name == "sandbox.createFileSystem") | .dur'
6378
7247
11850
13756
6555
7445
8487
15520
[...]
```

## Profile information {:#profile-information}

The profile contains multiple rows. Usually the bulk of rows represent Bazel
threads and their corresponding events, but some special rows are also included.

The special rows included depend on the version of Bazel invoked when the
profile was created, and may be customized by different flags.

Figure 1 shows a profile created with Bazel v5.3.1 and includes these rows:

*   `action count`: Displays how many concurrent actions were in flight. Click
    on it to see the actual value. Should go up to the value of
    [`--jobs`](/reference/command-line-reference#flag--jobs) in clean
    builds.
*   `CPU usage (Bazel)`: For each second of the build, displays the amount of
    CPU that was used by Bazel (a value of 1 equals one core being 100% busy).
*   `Critical Path`: Displays one block for each action on the critical path.
*   `Main Thread`: Bazel’s main thread. Useful to get a high-level picture of
    what Bazel is doing, for example "Launch Blaze", "evaluateTargetPatterns",
    and "runAnalysisPhase".
*   `Garbage Collector`: Displays minor and major Garbage Collection (GC)
    pauses.

## Common performance issues {:#common-performance-issues}

When analyzing performance profiles, look for:

*   Slower than expected analysis phase (`runAnalysisPhase`), especially on
    incremental builds. This can be a sign of a poor rule implementation, for
    example one that flattens depsets. Package loading can be slow by an
    excessive amount of targets, complex macros or recursive globs.
*   Individual slow actions, especially those on the critical path. It might be
    possible to split large actions into multiple smaller actions or reduce the
    set of (transitive) dependencies to speed them up. Also check for an unusual
    high non-`PROCESS_TIME` (such as `REMOTE_SETUP` or `FETCH`).
*   Bottlenecks, that is a small number of threads is busy while all others are
    idling / waiting for the result (see around 22s and 29s in Figure 1).
    Optimizing this will most likely require touching the rule implementations
    or Bazel itself to introduce more parallelism. This can also happen when
    there is an unusual amount of GC.

## Profile file format {:#profile-file-format}

The top-level object contains metadata (`otherData`) and the actual tracing data
(`traceEvents`). The metadata contains extra info, for example the invocation ID
and date of the Bazel invocation.

Example:

```json
{
  "otherData": {
    "build_id": "101bff9a-7243-4c1a-8503-9dc6ae4c3b05",
    "date": "Wed Oct 26 08:22:35 CEST 2022",
    "profile_finish_ts": "1677666095162000",
    "output_base": "/usr/local/google/_bazel_johndoe/573d4be77eaa72b91a3dfaa497bf8cd0"
  },
  "traceEvents": [
    {"name":"thread_name","ph":"M","pid":1,"tid":0,"args":{"name":"Critical Path"}},
    ...
    {"cat":"build phase marker","name":"Launch Blaze","ph":"X","ts":-1306000,"dur":1306000,"pid":1,"tid":21},
    ...
    {"cat":"package creation","name":"foo","ph":"X","ts":2685358,"dur":784,"pid":1,"tid":246},
    ...
    {"name":"thread_name","ph":"M","pid":1,"tid":11,"args":{"name":"Garbage Collector"}},
    {"cat":"gc notification","name":"minor GC","ph":"X","ts":825986,"dur":11000,"pid":1,"tid":11},
    ...
    {"cat":"action processing","name":"Compiling foo/bar.c","ph":"X","ts":54413389,"dur":357594,"pid":1,"args":{"mnemonic":"CppCompile"},"tid":341},
 ]
}
```

Timestamps (`ts`) and durations (`dur`) in the trace events are given in
microseconds. The category (`cat`) is one of enum values of `ProfilerTask`.
Note that some events are merged together if they are very short and close to
each other; pass
[`--noslim_json_profile`](/reference/command-line-reference#flag--slim_profile)
if you would like to prevent event merging.

See also the
[Chrome Trace Event Format Specification](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview){: .external}.

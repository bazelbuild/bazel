---
layout: documentation
title: Build Event Protocol
---

# Build Event Protocol

The [Build Event Protocol] allows third party programs to gain insight into a
Bazel invocation. For example, you could use the Build Event Protocol to gather
information for an IDE plugin or a dashboard that displays build results.

The protocol is a set of [protocol buffer] messages with some semantics defined
on top of it. It includes information about build and test results, build
progress, the build configuration and much more. The Build Event Protocol is
intended to be consumed programmatically and makes parsing Bazel’s command line
output a thing of the past.

## Contents

*  [Build Event Protocol Overview](#build-event-protocol-overview)
   *  [Build Event Graph](#build-event-graph)
   *  [The Build Event Protocol by Example](#the-build-event-protocol-by-example)
*  [Consuming the Build Event Protocol](#consuming-the-build-event-protocol)
   *  [Consume in a Binary Format](#consume-in-a-binary-format)
   *  [Consume in Text Formats](#consume-in-text-formats)
*  [The Build Event Service](#the-build-event-service)
   *  [Build Event Service Flags](#build-event-service-flags)
   *  [Authentication and Security](#authentication-and-security)

## Build Event Protocol Overview

The [Build Event Protocol] represents information about a build as events. A
build event is a protocol buffer message consisting of a build event identifier,
a set of child event identifiers, and a payload.

*  __Build Event Identifier:__ Depending on the kind of build event, it might be
an [opaque string] or [structured information] revealing more about the build
event. A build event identifier is unique within a build.

*  __Children:__ A build event may announce other build events, by including
their build event identifiers in its [children field]. For example, the
`PatternExpanded` build event announces the targets it expands to as children.
The protocol guarantees that all events, except for the first event, are
announced by a previous event.

* __Payload:__ The payload contains structured information about a build event,
encoded as a protocol buffer message specific to that event. Note, that the
payload might not be the expected type, but could be an `Aborted` message e.g.
if the build aborted prematurely.

### Build Event Graph
All build events form a directed acyclic graph through their parent and child
relationship. Every build event except for the initial build event has one or
more parent events. Please note that not all parent events of a child event must
necessarily be posted before it. When a build is complete (succeeded or failed)
all announced events will have been posted. In case of a Bazel crash or a failed
network transport, some announced build events may never be posted.

## The Build Event Protocol by Example
The full specification of the [Build Event Protocol] can be found in its
protocol buffer definition and describing it here is beyond the scope of this
document. However, it might be helpful to build up some intuition before looking
at the specification.

Consider a simple Bazel workspace that consists of two empty shell scripts
`foo.sh` and `foo_test.sh` and the following BUILD file:

```bash
sh_binary(
    name = "foo",
    srcs = ["foo.sh"],
)

sh_library(
    name = "foo_lib",
    data = [":foo"],
)

sh_test(
    name = "foo_test",
    srcs = ["foo_test.sh"],
    deps = [":foo_lib"],
)
```

When running `bazel test ...` on this project the build graph of the generated
build events will resemble the graph below. The arrows indicate the
aforementioned parent and child relationship. Note that some build events and
most fields have been omitted for brevity.

![bep-graph]

Initially, a `BuildStarted` event is published. The event informs us that the
build was invoked through the `bazel test` command and it also announces five
child events: `OptionsParsed`, `WorkspaceStatus`, `CommandLine`,
`PatternExpanded` and `Progress`. The first three events provide information
about how Bazel was invoked. The `PatternExpanded` build event provides insight
into which specific targets the `...` pattern expanded to: `//:foo`, `//:foo_lib`
and `//:foo_test`. It does so by declaring three `TargetConfigured` events as
children.

Note that the `TargetConfigured` event declares the `Configuration` event as a
child event, even though `Configuration` has been posted before the
`TargetConfigured` event.

Besides the parent and child relationship, events may also refer to each other
using their build event identifiers. For example, in the above graph the
`TargetComplete` event refers to the `NamedSetOfFiles` event in its `fileSets`
field.

Build events that refer to files (i.e. outputs) usually don’t embed the file
names and paths in the event. Instead, they contain the build event identifier
of a `NamedSetOfFiles` event, which will then contain the actual file names and
paths. The `NamedSetOfFiles` event allows a set of files to be reported once and
referred to by many targets. This structure is necessary because otherwise in
some cases the Build Event Protocol output size would grow quadratically with
the number of files. A `NamedSetOfFiles` event may also not have all its files
embedded, but instead refer to other `NamedSetOfFiles` events through their
build event identifiers.

Below is an instance of the `TargetComplete` event for the `//:foo_lib` target
from the above graph, printed in protocol buffer’s JSON representation. The
build event identifier contains the target as an opaque string and refers to the
`Configuration` event using its build event identifier. The event does not
announce any child events. The payload contains information about whether the
target was built successfully, the set of output files, and the kind of target
built.

```json
{
  "id": {
    "targetCompleted": {
      "label": "//:foo_lib",
      "configuration": {
        "id": "544e39a7f0abdb3efdd29d675a48bc6a"
      }
    }
  },
  "completed": {
    "success": true,
    "outputGroup": [{
      "name": "default",
      "fileSets": [{
        "id": "0"
      }]
    }],
    "targetKind": "sh_library rule"
  }
}
```

## Consuming the Build Event Protocol

### Consume in a Binary Format

To consume the Build Event Protocol in a binary format:

1. Have Bazel serialize the protocol buffer messages to a file by specifying the
option `--build_event_binary_file=/path/to/file`. The file will contain
serialized protocol buffer messages with each message being length delimited.
Each message is prefixed with its length encoded as a variable length integer.
This format can be read using the protocol buffer library’s
[parseDelimitedFrom(InputStream)] method.

2. Then, write a program that extracts the relevant information from the
serialized protocol buffer message.

### Consume in Text Formats

The following Bazel command line flags will output the Build Event Protocol in a
human-readable formats:

```
--build_event_text_file
--build_event_json_file
```

## The Build Event Service

The [Build Event Service] Protocol is a generic [gRPC] service for transmitting
build events. The Build Event Service protocol is independent of the Build Event
Protocol and treats the Build Event Protocol events as opaque bytes. Bazel ships
with a gRPC client implementation of the Build Event Service protocol that
transmits Build Event Protocol events. One can specify the endpoint to send the
events to using the `--bes_backend=HOST:PORT flag`. Bazel’s implementation also
supports TLS which can be enabled by specifying the `--tls_enabled flag`.

There is currently an experimental open source implementation of the [Build Event
Service](https://github.com/buildbarn/bb-event-service/) in Go as part of the 
Buildbarn suite of Remote Execution tools and services.

### Build Event Service Flags

Bazel has several flags related to the [Build Event Service] protocol, including:

*  `--bes_backend`
*  `--[no]bes_best_effort`
*  `--[no]bes_lifecycle_events`
*  `--bes_results_url`
*  `--bes_timeout`
*  `--project_id`

For a description of each of these flags, see the
[Command-Line Reference](command-line-reference.html).

### Authentication and Security

Bazel’s [Build Event Service] implementation also supports authentication and
TLS. These settings can be controlled using the below flags. Please note that
these flags are also used for Bazel’s Remote Execution. This implies that the
Build Event Service and Remote Execution Endpoints need to share the same
authentication and TLS infrastructure.

*  `--[no]google_default_credentials`
*  `--google_credentials`
*  `--google_auth_scopes`
*  `--tls_certificate`
*  `--[no]tls_enabled`

For a description of each of these flags, see the
[Command-Line Reference](command-line-reference.html).

### Build Event Service and Remote Caching

The BEP typically contains many references to log files (test.log, test.xml,
etc. ) stored on the machine where Bazel is running. A remote BES server
typically can't access these files as they are on different machines. A way
to work around this issue is to use Bazel with [remote caching]. Bazel will
upload all output files to the remote cache (including files referenced in
the BEP) and the BES server can then fetch the referenced files from the
cache.

See [GitHub issue 3689] for more details.

[remote caching]: https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/remote/README.md

[GitHub issue 3689]: https://github.com/bazelbuild/bazel/issues/3689

[Build Event Protocol]:
https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/buildeventstream/proto/build_event_stream.proto

[Build Event Service]:
https://github.com/googleapis/googleapis/blob/master/google/devtools/build/v1/publish_build_event.proto

[gRPC]: https://www.grpc.io

[protocol buffer]: https://developers.google.com/protocol-buffers/

[bep-graph]: /assets/bep-graph.svg

[parseDelimitedFrom(InputStream)]:
https://developers.google.com/protocol-buffers/docs/reference/java/com/google/protobuf/AbstractParser#parseDelimitedFrom-java.io.InputStream-

[opaque string]:
https://github.com/bazelbuild/bazel/blob/16a107d/src/main/java/com/google/devtools/build/lib/buildeventstream/proto/build_event_stream.proto#L91

[structured information]:
https://github.com/bazelbuild/bazel/blob/16a107d/src/main/java/com/google/devtools/build/lib/buildeventstream/proto/build_event_stream.proto#L123

[children field]:
https://github.com/bazelbuild/bazel/blob/16a107d/src/main/java/com/google/devtools/build/lib/buildeventstream/proto/build_event_stream.proto#L469

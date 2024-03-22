Project: /_project.yaml
Book: /_book.yaml

# Build Event Protocol

{% include "_buttons.html" %}

The [Build Event
Protocol](https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/buildeventstream/proto/build_event_stream.proto){: .external}
(BEP) allows third-party programs to gain insight into a Bazel invocation. For
example, you could use the BEP to gather information for an IDE
plugin or a dashboard that displays build results.

The protocol is a set of [protocol
buffer](https://developers.google.com/protocol-buffers/){: .external} messages with some
semantics defined on top of it. It includes information about build and test
results, build progress, the build configuration and much more. The BEP is
intended to be consumed programmatically and makes parsing Bazel’s
command line output a thing of the past.

The Build Event Protocol represents information about a build as events. A
build event is a protocol buffer message consisting of a build event identifier,
a set of child event identifiers, and a payload.

*  __Build Event Identifier:__ Depending on the kind of build event, it might be
an [opaque
string](https://github.com/bazelbuild/bazel/blob/7.1.0/src/main/java/com/google/devtools/build/lib/buildeventstream/proto/build_event_stream.proto#L131-L140){: .external}
or [structured
information](https://github.com/bazelbuild/bazel/blob/7.1.0/src/main/java/com/google/devtools/build/lib/buildeventstream/proto/build_event_stream.proto#L194-L205){: .external}
revealing more about the build event. A build event identifier is unique within
a build.

*  __Children:__ A build event may announce other build events, by including
their build event identifiers in its [children
field](https://github.com/bazelbuild/bazel/blob/7.1.0/src/main/java/com/google/devtools/build/lib/buildeventstream/proto/build_event_stream.proto#L1276){: .external}.
For example, the `PatternExpanded` build event announces the targets it expands
to as children. The protocol guarantees that all events, except for the first
event, are announced by a previous event.

* __Payload:__ The payload contains structured information about a build event,
encoded as a protocol buffer message specific to that event. Note that the
payload might not be the expected type, but could be an `Aborted` message
if the build aborted prematurely.

### Build event graph {:#build-event-graph}

All build events form a directed acyclic graph through their parent and child
relationship. Every build event except for the initial build event has one or
more parent events. Please note that not all parent events of a child event must
necessarily be posted before it. When a build is complete (succeeded or failed)
all announced events will have been posted. In case of a Bazel crash or a failed
network transport, some announced build events may never be posted.

The event graph's structure reflects the lifecycle of a command. Every BEP
graph has the following characteristic shape:

1. The root event is always a [`BuildStarted`](/remote/bep-glossary#buildstarted)
   event. All other events are its descendants.
1. Immediate children of the BuildStarted event contain metadata about the
   command.
1. Events containing data produced by the command, such as files built and test
   results, appear before the [`BuildFinished`](/remote/bep-glossary#buildfinished)
   event.
1. The [`BuildFinished`](/remote/bep-glossary#buildfinished) event *may* be followed
   by events containing summary information about the build (for example, metric
   or profiling data).

## Consuming Build Event Protocol {:#consuming-bep}

### Consume in binary format {:#consuming-bep-binary}

To consume the BEP in a binary format:

1. Have Bazel serialize the protocol buffer messages to a file by specifying the
   option `--build_event_binary_file=/path/to/file`. The file will contain
   serialized protocol buffer messages with each message being length delimited.
   Each message is prefixed with its length encoded as a variable length integer.
   This format can be read using the protocol buffer library’s
   [`parseDelimitedFrom(InputStream)`](https://developers.google.com/protocol-buffers/docs/reference/java/com/google/protobuf/AbstractParser#parseDelimitedFrom-java.io.InputStream-){: .external}
   method.

2. Then, write a program that extracts the relevant information from the
   serialized protocol buffer message.

### Consume in text or JSON formats {:#consuming-bep-text-json}

The following Bazel command line flags will output the BEP in
human-readable formats, such as text and JSON:

```
--build_event_text_file
--build_event_json_file
```

## Build Event Service {:#build-event-service}

The [Build Event
Service](https://github.com/googleapis/googleapis/blob/master/google/devtools/build/v1/publish_build_event.proto){: .external}
Protocol is a generic [gRPC](https://www.grpc.io){: .external} service for publishing build events. The Build Event
Service protocol is independent of the BEP and treats BEP events as opaque bytes.
Bazel ships with a gRPC client implementation of the Build Event Service protocol that
publishes Build Event Protocol events. One can specify the endpoint to send the
events to using the `--bes_backend=HOST:PORT` flag. If your backend uses gRPC,
you must prefix the address with the appropriate scheme: `grpc://` for plaintext
gRPC and `grpcs://` for gRPC with TLS enabled.

### Build Event Service flags {:#bes-flags}

Bazel has several flags related to the Build Event Service protocol, including:

*  `--bes_backend`
*  `--[no]bes_best_effort`
*  `--[no]bes_lifecycle_events`
*  `--bes_results_url`
*  `--bes_timeout`
*  `--bes_instance_name`

For a description of each of these flags, see the
[Command-Line Reference](/reference/command-line-reference).

### Authentication and security {:#authentication-security}

Bazel’s Build Event Service implementation also supports authentication and TLS.
These settings can be controlled using the below flags. Please note that these
flags are also used for Bazel’s Remote Execution. This implies that the Build
Event Service and Remote Execution Endpoints need to share the same
authentication and TLS infrastructure.

*  `--[no]google_default_credentials`
*  `--google_credentials`
*  `--google_auth_scopes`
*  `--tls_certificate`
*  `--[no]tls_enabled`

For a description of each of these flags, see the
[Command-Line Reference](/reference/command-line-reference).

### Build Event Service and remote caching {:#bes-remote-caching}

The BEP typically contains many references to log files (test.log, test.xml,
etc. ) stored on the machine where Bazel is running. A remote BES server
typically can't access these files as they are on different machines. A way to
work around this issue is to use Bazel with [remote
caching](/remote/caching).
Bazel will upload all output files to the remote cache (including files
referenced in the BEP) and the BES server can then fetch the referenced files
from the cache.

See [GitHub issue 3689](https://github.com/bazelbuild/bazel/issues/3689){: .external} for
more details.

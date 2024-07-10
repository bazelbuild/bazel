Project: /_project.yaml
Book: /_book.yaml

# Creating Persistent Workers

{% include "_buttons.html" %}

[Persistent workers](/remote/persistent) can make your build faster. If
you have repeated actions in your build that have a high startup cost or would
benefit from cross-action caching, you may want to implement your own persistent
worker to perform these actions.

The Bazel server communicates with the worker using `stdin`/`stdout`. It
supports the use of protocol buffers or JSON strings.

The worker implementation has two parts:

*   The [worker](#making-worker).
*   The [rule that uses the worker](#rule-uses-worker).

## Making the worker {:#making-worker}

A persistent worker upholds a few requirements:

*   It reads
    [WorkRequests](https://github.com/bazelbuild/bazel/blob/54a547f30fd582933889b961df1d6e37a3e33d85/src/main/protobuf/worker_protocol.proto#L36){: .external}
    from its `stdin`.
*   It writes
    [WorkResponses](https://github.com/bazelbuild/bazel/blob/54a547f30fd582933889b961df1d6e37a3e33d85/src/main/protobuf/worker_protocol.proto#L77){: .external}
    (and only `WorkResponse`s) to its `stdout`.
*   It accepts the `--persistent_worker` flag. The wrapper must recognize the
    `--persistent_worker` command-line flag and only make itself persistent if
    that flag is passed, otherwise it must do a one-shot compilation and exit.

If your program upholds these requirements, it can be used as a persistent
worker!

### Work requests {:#work-requests}

A `WorkRequest` contains a list of arguments to the worker, a list of
path-digest pairs representing the inputs the worker can access (this isn’t
enforced, but you can use this info for caching), and a request id, which is 0
for singleplex workers.

NOTE: While the protocol buffer specification uses "snake case" (`request_id`),
the JSON protocol uses "camel case" (`requestId`). This document uses camel case
in the JSON examples, but snake case when talking about the field regardless of
protocol.

```json
{
  "arguments" : ["--some_argument"],
  "inputs" : [
    { "path": "/path/to/my/file/1", "digest": "fdk3e2ml23d"},
    { "path": "/path/to/my/file/2", "digest": "1fwqd4qdd" }
 ],
  "requestId" : 12
}
```

The optional `verbosity` field can be used to request extra debugging output
from the worker. It is entirely up to the worker what and how to output. Higher
values indicate more verbose output. Passing the `--worker_verbose` flag to
Bazel sets the `verbosity` field to 10, but smaller or larger values can be used
manually for different amounts of output.

The optional `sandbox_dir` field is used only by workers that support
[multiplex sandboxing](/remote/multiplex).

### Work responses {:#work-responses}

A `WorkResponse` contains a request id, a zero or nonzero exit code, and an
output message describing any errors encountered in processing or executing
the request. A worker should capture the `stdout` and `stderr` of any tool it
calls and report them through the `WorkResponse`. Writing it to the `stdout` of
the worker process is unsafe, as it will interfere with the worker protocol.
Writing it to the `stderr` of the worker process is safe, but the result is
collected in a per-worker log file instead of ascribed to individual actions.

```json
{
  "exitCode" : 1,
  "output" : "Action failed with the following message:\nCould not find input
    file \"/path/to/my/file/1\"",
  "requestId" : 12
}
```

As per the norm for protobufs, all fields are optional. However, Bazel requires
the `WorkRequest` and the corresponding `WorkResponse`, to have the same request
id, so the request id must be specified if it is nonzero. This is a valid
`WorkResponse`.

```json
{
  "requestId" : 12,
}
```

A `request_id` of 0 indicates a "singleplex" request, used when this request
cannot be processed in parallel with other requests. The server guarantees that
a given worker receives requests with either only `request_id` 0 or only
`request_id` greater than zero. Singleplex requests are sent in serial, for
example if the server doesn't send another request until it has received a
response (except for cancel requests, see below).

**Notes**

*   Each protocol buffer is preceded by its length in `varint` format (see
    [`MessageLite.writeDelimitedTo()`](https://developers.google.com/protocol-buffers/docs/reference/java/com/google/protobuf/MessageLite.html#writeDelimitedTo-java.io.OutputStream-){: .external}.
*   JSON requests and responses are not preceded by a size indicator.
*   JSON requests uphold the same structure as the protobuf, but use standard
    JSON and use camel case for all field names.
*   In order to maintain the same backward and forward compatibility properties
    as protobuf, JSON workers must tolerate unknown fields in these messages,
    and use the protobuf defaults for missing values.
*   Bazel stores requests as protobufs and converts them to JSON using
    [protobuf's JSON format](https://cs.opensource.google/protobuf/protobuf/+/master:java/util/src/main/java/com/google/protobuf/util/JsonFormat.java)

### Cancellation {:#cancellation}

Workers can optionally allow work requests to be cancelled before they finish.
This is particularly useful in connection with dynamic execution, where local
execution can regularly be interrupted by a faster remote execution. To allow
cancellation, add `supports-worker-cancellation: 1` to the
`execution-requirements` field (see below) and set the
`--experimental_worker_cancellation` flag.

A **cancel request** is a `WorkRequest` with the `cancel` field set (and
similarly a **cancel response** is a `WorkResponse` with the `was_cancelled`
field set). The only other field that must be in a cancel request or cancel
response is `request_id`, indicating which request to cancel. The `request_id`
field will be 0 for singleplex workers or the non-0 `request_id` of a previously
sent `WorkRequest` for multiplex workers. The server may send cancel requests
for requests that the worker has already responded to, in which case the cancel
request must be ignored.

Each non-cancel `WorkRequest` message must be answered exactly once, whether or
not it was cancelled. Once the server has sent a cancel request, the worker may
respond with a `WorkResponse` with the `request_id` set and the `was_cancelled`
field set to true. Sending a regular `WorkResponse` is also accepted, but the
`output` and `exit_code` fields will be ignored.

Once a response has been sent for a `WorkRequest`, the worker must not touch the
files in its working directory. The server is free to clean up the files,
including temporary files.

## Making the rule that uses the worker {:#rule-uses-worker}

You'll also need to create a rule that generates actions to be performed by the
worker. Making a Starlark rule that uses a worker is just like
[creating any other rule](https://github.com/bazelbuild/examples/tree/master/rules){: .external}.

In addition, the rule needs to contain a reference to the worker itself, and
there are some requirements for the actions it produces.

### Referring to the worker

The rule that uses the worker needs to contain a field that refers to the worker
itself, so you'll need to create an instance of a `\*\_binary` rule to define
your worker. If your worker is called `MyWorker.Java`, this might be the
associated rule:

```python
java_binary(
    name = "worker",
    srcs = ["MyWorker.Java"],
)
```

This creates the "worker" label, which refers to the worker binary. You'll then
define a rule that *uses* the worker. This rule should define an attribute that
refers to the worker binary.

If the worker binary you built is in a package named "work", which is at the top
level of the build, this might be the attribute definition:

```python
"worker": attr.label(
    default = Label("//work:worker"),
    executable = True,
    cfg = "exec",
)
```

`cfg = "exec"` indicates that the worker should be built to run on your
execution platform rather than on the target platform (i.e., the worker is used
as tool during the build).

### Work action requirements {:#work-action-requirements}

The rule that uses the worker creates actions for the worker to perform. These
actions have a couple of requirements.

*   The *"arguments"* field. This takes a list of strings, all but the last of
    which are arguments passed to the worker upon startup. The last element in
    the "arguments" list is a `flag-file` (@-preceded) argument. Workers read
    the arguments from the specified flagfile on a per-WorkRequest basis. Your
    rule can write non-startup arguments for the worker to this flagfile.

*   The *"execution-requirements"* field, which takes a dictionary containing
    `"supports-workers" : "1"`, `"supports-multiplex-workers" : "1"`, or both.

    The "arguments" and "execution-requirements" fields are required for all
    actions sent to workers. Additionally, actions that should be executed by
    JSON workers need to include `"requires-worker-protocol" : "json"` in the
    execution requirements field. `"requires-worker-protocol" : "proto"` is also
    a valid execution requirement, though it’s not required for proto workers,
    since they are the default.

    You can also set a `worker-key-mnemonic` in the execution requirements. This
    may be useful if you're reusing the executable for multiple action types and
    want to distinguish actions by this worker.

*   Temporary files generated in the course of the action should be saved to the
    worker's directory. This enables sandboxing.

Note: To pass an argument starting with a literal `@`, start the argument with
`@@` instead. If an argument is also an external repository label, it will not
be considered a flagfile argument.

Assuming a rule definition with "worker" attribute described above, in addition
to a "srcs" attribute representing the inputs, an "output" attribute
representing the outputs, and an "args" attribute representing the worker
startup args, the call to `ctx.actions.run` might be:

```python
ctx.actions.run(
  inputs=ctx.files.srcs,
  outputs=[ctx.outputs.output],
  executable=ctx.executable.worker,
  mnemonic="someMnemonic",
  execution_requirements={
    "supports-workers" : "1",
    "requires-worker-protocol" : "json"},
  arguments=ctx.attr.args + ["@flagfile"]
 )
```

For another example, see
[Implementing persistent workers](/remote/persistent#implementation).

## Examples {:#examples}

The Bazel code base uses
[Java compiler workers](https://github.com/bazelbuild/bazel/blob/a4251eab6988d6cf4f5e35681fbe2c1b0abe48ef/src/java_tools/buildjar/java/com/google/devtools/build/buildjar/BazelJavaBuilder.java){: .external},
in addition to an
[example JSON worker](https://github.com/bazelbuild/bazel/blob/c65f768fec9889bbf1ee934c61d0dc061ea54ca2/src/test/java/com/google/devtools/build/lib/worker/ExampleWorker.java){: .external}
that is used in our integration tests.

You can use their
[scaffolding](https://github.com/bazelbuild/bazel/blob/a4251eab6988d6cf4f5e35681fbe2c1b0abe48ef/src/main/java/com/google/devtools/build/lib/worker/WorkRequestHandler.java){: .external}
to make any Java-based tool into a worker by passing in the correct callback.

For an example of a rule that uses a worker, take a look at Bazel's
[worker integration test](https://github.com/bazelbuild/bazel/blob/22b4dbcaf05756d506de346728db3846da56b775/src/test/shell/integration/bazel_worker_test.sh#L106){: .external}.

External contributors have implemented workers in a variety of languages; take a
look at
[Polyglot implementations of Bazel persistent workers](https://github.com/Ubehebe/bazel-worker-examples){: .external}.
You can
[find many more examples on GitHub](https://github.com/search?q=bazel+workrequest&type=Code){: .external}!

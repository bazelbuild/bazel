---
layout: documentation
title: Creating persistent workers
---

# Creating persistent workers

[Persistent workers](persistent-workers.html) can make your build faster.
If you have repeated actions in your build that have a high startup cost or
would benefit from cross-action caching, you may want to implement your own
persistent worker to perform these actions.

The worker implementation has two parts:

* The [worker](#making-the-worker),
* The [rule that uses the worker](#making-the-rule-that-uses-the-worker).

## Making the worker

A worker upholds a few requirements:

* It reads [WorkRequests](https://github.com/bazelbuild/bazel/blob/6d1b9725b1e201ca3f25d8ec2a730a20aab62c6e/src/main/protobuf/worker_protocol.proto#L35)
from its `stdin`.
* It writes [WorkResponses](https://github.com/bazelbuild/bazel/blob/6d1b9725b1e201ca3f25d8ec2a730a20aab62c6e/src/main/protobuf/worker_protocol.proto#L49)
(and only `WorkResponse`s) to its `stdout`.
* It accepts the `--persistent_worker` flag.

If your program upholds these requirements, it can be used as a worker!

### Work requests

A `WorkRequest` contains a list of arguments to the worker, a list of path-digest
pairs representing the inputs the worker can access (this isn’t enforced, but
you can use this info for caching), and a request id, which is 0 for singleplex
workers.

```json
{
  “args” : [“--some_argument”],
  “inputs” : [
    { “/path/to/my/file/1” : “fdk3e2ml23d”},
    { “/path/to/my/file/2” : “1fwqd4qdd” }
 ],
  “request_id” : 12
}
```

### Work responses

A `WorkResponse` should contain the same request id, a zero or nonzero exit
code, and an output string that contains any errors encountered in processing
or executing the request. Workers may write additional output to `stderr`, but
they must only write `WorkResponse`s to `stdout`.

```json
{
  “exit_code” : 1,
  “output” : “Action failed with the following message:\nCould not find input
    file “/path/to/my/file/1”,
  “request_id” : 12
}
```

As per the norm for protobufs, the fields are optional. However, Bazel requires
the `WorkRequest` and the corresponding `WorkResponse`, to have the same request
id, so the request id must be specified if it is nonzero. This is a valid
`WorkResponse`.

```json
{
  “request_id” : 12,
}
```

## Making the rule that uses the worker

You'll also need to create a rule that generates actions to be performed by the
worker. Making a Starlark rule that uses a worker is just like [creating any other rule](https://github.com/bazelbuild/examples/tree/master/rules).

In addition, the rule needs to contain a reference to the worker itself, and
there are some requirements for the actions it produces.

### Referring to the worker
The rule that uses the worker needs to contain a field that refers to the worker itself,
so you'll need to create an instance of a `\*\_binary` rule to define your
worker. If your worker is called `MyWorker.Java`, this might be the associated
rule:

```python
java_binary(
    name = “worker”,
    srcs = [“MyWorker.Java”],
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
    cfg = "host",
)
```

`cfg = "host"` indicates that the worker should be built to run on your host
platform.

### Work action requirements

The rule that uses the worker creates actions for the worker to perform. These
actions have a couple of requirements.


* The _“arguments”_ field. This takes a list of strings, all but the last
  of which are arguments passed to the worker upon startup. The last element in
  the “arguments” list is a `flag-file` (@-preceded) argument. Workers read
  the arguments from the specified flagfile on a per-WorkRequest basis. Your
  rule can write non-startup arguments for the worker to this flagfile.

* The _“execution-requirements”_ field, which takes a dictionary containing
  `“supports-workers” : “1”`, `“supports-multiplex-workers” : “1”`, or both.

  The "arguments" and "execution-requirements" fields are required for all
  actions sent to workers. Additionally, actions that should be executed by
  JSON workers need to include `“requires-worker-protocol” : “json”` in the
  execution requirements field. `“requires-worker-protocol” : “proto”` is also
  a valid execution requirement, though it’s not required for proto workers,
  since they are the default.

  You can also set a "worker-key-mnemonic" in the execution requirements. This
  may be useful if you're reusing the executable for multiple action types and
  want to distinguish actions by this worker.

* Temporary files generated in the course of the action should be saved to the
  worker's directory. This enables sandboxing.

Assuming a rule definition with "worker" attribute described above, in addition
to a "srcs" attribute representing the inputs, an "output" attribute
representing the outputs, and an "args" attribute representing the worker
startup args, the call to `ctx.actions.run` might be:

```python
ctx.actions.run(
  inputs=ctx.files.srcs,
  outputs=[ctx.attr.output],
  executable=ctx.attr.worker,
  mnemonic="someMnemonic",
  execution_requirements={
    “supports-workers” : “1”,
    “requires-worker-protocol” : “json},
  arguments=ctx.attr.args + [“@flagfile”]
 )
```
## Examples

The Bazel code base uses [Java compiler workers](https://github.com/bazelbuild/bazel/blob/a4251eab6988d6cf4f5e35681fbe2c1b0abe48ef/src/java_tools/buildjar/java/com/google/devtools/build/buildjar/BazelJavaBuilder.java),
in addition to an [example JSON worker](https://github.com/bazelbuild/bazel/blob/c65f768fec9889bbf1ee934c61d0dc061ea54ca2/src/test/java/com/google/devtools/build/lib/worker/ExampleWorker.java) that is used in our integration tests.

You can use their [scaffolding](https://github.com/bazelbuild/bazel/blob/a4251eab6988d6cf4f5e35681fbe2c1b0abe48ef/src/main/java/com/google/devtools/build/lib/worker/WorkRequestHandler.java) to make any Java-based tool into a worker by passing in the correct
callback.

For an example of a rule that uses a worker, take a look at Bazel's
[worker integration test](https://github.com/bazelbuild/bazel/blob/22b4dbcaf05756d506de346728db3846da56b775/src/test/shell/integration/bazel_worker_test.sh#L106).

External contributors have implemented workers in a variety of languages; you
can [find many more examples on GitHub](https://github.com/search?q=bazel+workrequest&type=Code)!

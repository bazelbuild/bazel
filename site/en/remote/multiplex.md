Project: /_project.yaml
Book: /_book.yaml

# Multiplex Workers (Experimental Feature)

{% include "_buttons.html" %}

This page describes multiplex workers, how to write multiplex-compatible
rules, and workarounds for certain limitations.

Caution: Experimental features are subject to change at any time.

_Multiplex workers_ allow Bazel to handle multiple requests with a single worker
process. For multi-threaded workers, Bazel can use fewer resources to
achieve the same, or better performance. For example, instead of having one
worker process per worker, Bazel can have four multiplexed workers talking to
the same worker process, which can then handle requests in parallel. For
languages like Java and Scala, this saves JVM warm-up time and JIT compilation
time, and in general it allows using one shared cache between all workers of
the same type.

## Overview {:#overview}

There are two layers between the Bazel server and the worker process. For certain
mnemonics that can run processes in parallel, Bazel gets a `WorkerProxy` from
the worker pool. The `WorkerProxy` forwards requests to the worker process
sequentially along with a `request_id`, the worker process processes the request
and sends responses to the `WorkerMultiplexer`. When the `WorkerMultiplexer`
receives a response, it parses the `request_id` and then forwards the responses
back to the correct `WorkerProxy`. Just as with non-multiplexed workers, all
communication is done over standard in/out, but the tool cannot just use
`stderr` for user-visible output ([see below](#output)).

Each worker has a key. Bazel uses the key's hash code (composed of environment
variables, the execution root, and the mnemonic) to determine which
`WorkerMultiplexer` to use. `WorkerProxy`s communicate with the same
`WorkerMultiplexer` if they have the same hash code. Therefore, assuming
environment variables and the execution root are the same in a single Bazel
invocation, each unique mnemonic can only have one `WorkerMultiplexer` and one
worker process. The total number of workers, including regular workers and
`WorkerProxy`s, is still limited by `--worker_max_instances`.

## Writing multiplex-compatible rules {:#multiplex-rules}

The rule's worker process should be multi-threaded to take advantage of
multiplex workers. Protobuf allows a ruleset to parse a single request even
though there might be multiple requests piling up in the stream. Whenever the
worker process parses a request from the stream, it should handle the request in
a new thread. Because different thread could complete and write to the stream at
the same time, the worker process needs to make sure the responses are written
atomically (messages don't overlap). Responses must contain the
`request_id` of the request they're handling.

### Handling multiplex output {:#output}

Multiplex workers need to be more careful about handling their output than
singleplex workers. Anything sent to `stderr` will go into a single log file
shared among all `WorkerProxy`s of the same type,
randomly interleaved between concurrent requests. While redirecting `stdout`
into `stderr` is a good idea, do not collect that output into the `output`
field of `WorkResponse`, as that could show the user mangled pieces of output.
If your tool only sends user-oriented output to `stdout` or `stderr`, you will
need to change that behaviour before you can enable multiplex workers.

## Enabling multiplex workers {:#multiplex-workers}

Multiplex workers are not enabled by default. A ruleset can turn on multiplex
workers by using the `supports-multiplex-workers` tag in the
`execution_requirements` of an action (just like the `supports-workers` tag
enables regular workers). As is the case when using regular workers, a worker
strategy needs to be specified, either at the ruleset level (for example,
`--strategy=[some_mnemonic]=worker`) or generally at the strategy level (for
example, `--dynamic_local_strategy=worker,standalone`.) No additional flags are
necessary, and `supports-multiplex-workers` takes precedence over
`supports-workers`, if both are set. You can turn off multiplex workers
globally by passing `--noworker_multiplex`.

A ruleset is encouraged to use multiplex workers if possible,  to reduce memory
pressure and improve performance. However, multiplex workers are not currently
compatible with [dynamic execution](/remote/dynamic) unless they
implement multiplex sandboxing. Attempting to run non-sandboxed multiplex
workers with dynamic execution will silently use sandboxed
singleplex workers instead.

## Multiplex sandboxing

Multiplex workers can be sandboxed by adding explicit support for it in the
worker implementations. While singleplex worker sandboxing can be done by
running each worker process in its own sandbox, multiplex workers share the
process working directory between multiple parallel requests. To allow
sandboxing of multiplex workers, the worker must support reading from and
writing to a subdirectory specified in each request, instead of directly in
its working directory.

To support multiplex sandboxing, the worker must use the `sandbox_dir` field
from the `WorkRequest` and use that as a prefix for all file reads and writes.
While the `arguments` and `inputs` fields remain unchanged from an unsandboxed
request, the actual inputs are relative to the `sandbox_dir`. The worker must
translate file paths found in `arguments` and `inputs` to read from this
modified path, and must also write all outputs relative to the `sandbox_dir`.
This includes paths such as '.', as well as paths found in files specified
in the arguments (such as ["argfile"](https://docs.oracle.com/javase/7/docs/technotes/tools/windows/javac.html#commandlineargfile) arguments).

Once a worker supports multiplex sandboxing, the ruleset can declare this
support by adding `supports-multiplex-sandboxing` to the
`execution_requirements` of an action. Bazel will then use multiplex sandboxing
if the `--experimental_worker_multiplex_sandboxing` flag is passed, or if
the worker is used with dynamic execution.

The worker files of a sandboxed multiplex worker are still relative to the
working directory of the worker process. Thus, if a file is
used both for running the worker and as an input, it must be specified both as
an input in the flagfile argument as well as in `tools`, `executable`, or
`runfiles`.

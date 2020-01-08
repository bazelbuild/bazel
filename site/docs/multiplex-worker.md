---
layout: documentation
title: Multiplex Workers
---

# Multiplex Workers (experimental feature)

Multiplex Workers allow Bazel to handle multiple requests with a single worker
process. For multi-threaded workers, Bazel is able to use less resources to
achieve the same, or even better performance. For example, instead of having one
worker process per worker, Bazel can have four multiplexed workers talking to
the same worker process, which can then handle requests in parallel. For
languages like Java and Scala, this saves JVM warm-up time and JIT compilation
time.

## Contents

*  [Multiplex Workers Overview](#multiplex-workers-overview)
*  [A Guide to Write a Multiplex-Compatible Ruleset](#a-guide-to-write-a-multiplex-compatible-ruleset)
*  [Enable Multiplex Workers](#enable-multiplex-workers)
   *  [WARNING](#warning)

## Multiplex Workers Overview

We add two layers between the Bazel server and the worker process. For certain
mnemonics that can run processes in parallel, Bazel gets a `WorkerProxy` from
the worker pool. The `WorkerProxy` forwards requests to the worker process
sequentially along with a `request_id`, the worker process processes the request
and sends responses to the `WorkerMultiplexer`. When the `WorkerMultiplexer`
receives a response, it parses the `request_id` and then forwards the responses
back to the correct `WorkerProxy`. Just as with non-multiplexed workers, all
communication is done over standard in/out.

Each worker has a key. Bazel uses the hash code (composed of environment
variables, the execution root, and the mnemonic) of the key to determine which
`WorkerMultiplexer` to use. `WorkerProxy`s communicate with the same
`WorkerMultiplexer` if they have the same hash code. Therefore, assuming
environment variables and the execution root are the same in a single Bazel
invocation, each unique mnemonic can only have one `WorkerMultiplexer` and one
worker process. The total number of workers, including regular workers and
`WorkerProxy`s, is still limited by `--worker_max_instances`.

## A Guide to Write a Multiplex-Compatible Ruleset

The rule's worker process should be multi-threaded to take advantage of
Multiplex Workers. Protobuf allows a ruleset to parse a single request even
though there might be multiple requests piling up in the stream. Whenever the
worker process parses a request from the stream, it should handle the request in
a new thread. Since different thread could complete and write to the stream at
the same time, the worker process needs to make sure the responses are written
atomically (i.e. messages don't overlap). Responses must contain the
`request_id` of the request they're handling.

## Enable Multiplex Workers

Multiplex workers are not enabled by default. A ruleset can turn on Multiplex
Workers by using the `supports-multiplex-workers` tag in the
`execution_requirements` of an action (just like the `supports-workers` tag
enables regular workers). As is the case when using regular workers, a worker
strategy needs to be specified, either at the ruleset level, eg.
`--strategy=[some_mnemonic]=worker`, or generally at the strategy level, eg.
`--dynamic_local_strategy=worker,standalone`. No additional flags are
necessary, and `supports-multiplex-workers` takes precedence over
`supports-workers`, if both are set. A ruleset is encouraged to use Multiplex
Workers if possible, since this will improve performance.

### WARNING

Due to a rare bug, Multiplex Workers are currently unstable. Occasionally, you
might see Bazel hanging at the execution phase. We believe this happens because
Multiplex Workers are waiting for responses from the worker process which never
comes. Bazel will hang indefinitely. If you see this behavior, stop the Bazel
server and rerun. It is not expected to happen often. We are actively working on
a fix.

There is also a chance that the issue lies in the ruleset worker implementation.
It is possible a thread dies or a race condition occurs. Make sure the worker
process always returns responses in all circumstances.

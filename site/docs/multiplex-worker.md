---
layout: documentation
title: Multiplex workers
category: extending
---

# Multiplex Workers (Experimental Feature)

This page describes multiplex workers, how to write multiplex-compatible
rules, and workarounds for certain limitations.

**Caution:** Experimental features are subject to change at any time.

_Multiplex workers_ allow Bazel to handle multiple requests with a single worker
process. For multi-threaded workers, Bazel can use fewer resources to
achieve the same, or better performance. For example, instead of having one
worker process per worker, Bazel can have four multiplexed workers talking to
the same worker process, which can then handle requests in parallel. For
languages like Java and Scala, this saves JVM warm-up time and JIT compilation
time.

## Overview

There are two layers between the Bazel server and the worker process. For certain
mnemonics that can run processes in parallel, Bazel gets a `WorkerProxy` from
the worker pool. The `WorkerProxy` forwards requests to the worker process
sequentially along with a `request_id`, the worker process processes the request
and sends responses to the `WorkerMultiplexer`. When the `WorkerMultiplexer`
receives a response, it parses the `request_id` and then forwards the responses
back to the correct `WorkerProxy`. Just as with non-multiplexed workers, all
communication is done over standard in/out.

Each worker has a key. Bazel uses the key's hash code (composed of environment
variables, the execution root, and the mnemonic) to determine which
`WorkerMultiplexer` to use. `WorkerProxy`s communicate with the same
`WorkerMultiplexer` if they have the same hash code. Therefore, assuming
environment variables and the execution root are the same in a single Bazel
invocation, each unique mnemonic can only have one `WorkerMultiplexer` and one
worker process. The total number of workers, including regular workers and
`WorkerProxy`s, is still limited by `--worker_max_instances`.

## Writing multiplex-compatible rules

The rule's worker process should be multi-threaded to take advantage of
multiplex workers. Protobuf allows a ruleset to parse a single request even
though there might be multiple requests piling up in the stream. Whenever the
worker process parses a request from the stream, it should handle the request in
a new thread. Because different thread could complete and write to the stream at
the same time, the worker process needs to make sure the responses are written
atomically (i.e. messages don't overlap). Responses must contain the
`request_id` of the request they're handling.

## Enabling multiplex workers

Multiplex workers are not enabled by default. A ruleset can turn on multiplex
workers by using the `supports-multiplex-workers` tag in the
`execution_requirements` of an action (just like the `supports-workers` tag
enables regular workers). As is the case when using regular workers, a worker
strategy needs to be specified, either at the ruleset level (for example,
`--strategy=[some_mnemonic]=worker`) or generally at the strategy level (for
example, `--dynamic_local_strategy=worker,standalone`.) No additional flags are
necessary, and `supports-multiplex-workers` takes precedence over
`supports-workers`, if both are set. A ruleset is encouraged to use multiplex
workers if possible, to improve performance.

### Warning about rare bug

Due to a rare bug, multiplex workers are currently unstable. Occasionally,
Bazel hangs indefinitely at the execution phase. If you see this behavior,
stop the Bazel server and rerun. This delay is probably caused by

 * Multiplex workers waiting for responses from the worker process that never
   comes.
 * Incorrectly configured ruleset worker implementation where a thread dies or
   a race condition occurs. To counteract this, ensure the worker process
   returns responses in all circumstances.

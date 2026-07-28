# Local Jobserver

When building with `--experimental_local_jobserver`, tools that respect the
[GNU make jobserver protocol](https://www.gnu.org/software/make/manual/html_node/Job-Slots.html)
can coordinate with Bazel to have the same view of available CPU slots, avoiding
CPU overloading.

## Background

Bazel’s execution model statically assigns CPU resources (usually 1 core) to an
action a priori. However, some tools are capable of using multiple cores, either
to a fixed degree specified by a parameter, as in `pigz -p 4` or `make -j
$(nproc)`, or dynamically at run-time. The static use case is accommodated by
resource sets and execution requirements, but there is no facility for allowing
actions to make just-in-time decisions about their own parallelism and
communicate them back to Bazel. As a result, actions that decide to spawn
multiple threads or subprocesses do so without regard to anything else Bazel may
be running at the time, resulting in CPU oversubscription. The jobserver is an
answer to this problem.

[Originally designed to solve the same problem among recursive `make`
invocations](https://make.mad-scientist.net/papers/jobserver-implementation/), a
“jobserver” is nothing more than a synchronization primitive that holds a number
of “tokens,” one per CPU core. Jobs consume tokens to “reserve” cores, do their
work, and write the tokens back when they are done. The primitive is a FIFO on
POSIX, and a
[semaphore](https://learn.microsoft.com/en-us/windows/win32/sync/semaphore-objects)
on Windows.

## When to use

This feature is intended to ease pressure on a machine when running heavy,
self-parallelizing jobs. Two common Bazel use cases are
[rules_rust](https://github.com/bazelbuild/rules_rust) and
[rules_foreign_cc](https://github.com/bazel-contrib/rules_foreign_cc), since the
[Rust compiler](https://doc.rust-lang.org/rustc/jobserver.html) and `make` both
support the jobserver protocol and can do CPU- and RAM-intensive activities. Use
it if your machine becomes unresponsive during builds. It is not intended to
make your build _faster_—wall time may indeed increase—but the load throughout
the build will be more in line with the number of CPUs you have.

## How to use

Tag actions or rules[^1] with `supports-jobserver` and `MAKEFLAGS` will be
injected into their environment with an appropriate value for
`--jobserver-auth`. Note that this is not part of the action cache key, so the
output should not depend on whether this feature is used or how many tokens are
available.

## Architecture

At execution setup, `ExecutionTool` configures the process-wide `LocalJobserver`
with an OS-appropriate backend. This class starts a manager thread, whose job is
to constantly resize the jobserver’s token pool to reflect CPUs that are
available for use, i.e. not consumed by other Bazel actions, and report back to
`ResourceManager` the number of tokens held by jobserver-aware tools.[^2] This
two-way communication is the essence of the feature, as it allows Bazel and the
jobserver to share their respective CPU consumption with each other.

### Per-tick accounting

Accounting is the same technique on all platforms: completely drain the pool,
count how many tokens were acquired, and write back only the amount to keep. We
cannot “peek” how many tokens are in the pool because

- on Windows, there is no documented function to query a semaphore in a
  non-mutating way
- on macOS, `FileInputStream.available()` is broken on FIFOs (always returns 0)

The number of tokens kept in the pool is

$$max(0, \lfloor totalCpuCapacity - cpuReservedByActions \rfloor - tokensHeldByTools)$$

## Caveats

- `make` clients must be version 4.4 or later. Prior versions do not support the
  `fifo` style and will abort.
- `output_base` paths containing whitespace are rejected on POSIX.

[^1]: Requires `--incompatible_allow_tags_propagation`, which is the default.
[^2]: When using `--experimental_cpu_load_scheduling`, measured CPU load already
includes the clients’ parallel work, so nothing is explicitly added.

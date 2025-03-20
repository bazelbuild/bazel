Project: /_project.yaml
Book: /_book.yaml

# Sandboxing

{% include "_buttons.html" %}

This article covers sandboxing in Bazel and debugging your sandboxing
environment.

*Sandboxing* is a permission restricting strategy that isolates processes from
each other or from resources in a system. For Bazel, this means restricting file
system access.

Bazel's file system sandbox runs processes in a working directory that only
contains known inputs, such that compilers and other tools don't see source
files they should not access, unless they know the absolute paths to them.

Sandboxing doesn't hide the host environment in any way. Processes can freely
access all files on the file system. However, on platforms that support user
namespaces, processes can't modify any files outside their working directory.
This ensures that the build graph doesn't have hidden dependencies that could
affect the reproducibility of the build.

More specifically, Bazel constructs an `execroot/` directory for each action,
which acts as the action's work directory at execution time. `execroot/`
contains all input files to the action and serves as the container for any
generated outputs. Bazel then uses an operating-system-provided technique,
containers on Linux and `sandbox-exec` on macOS, to constrain the action within
`execroot/`.

## Reasons for sandboxing {:#sandboxing-reasons}

-   Without action sandboxing, Bazel doesn't know if a tool uses undeclared
    input files (files that are not explicitly listed in the dependencies of an
    action). When one of the undeclared input files changes, Bazel still
    believes that the build is up-to-date and won’t rebuild the action. This can
    result in an incorrect incremental build.

-   Incorrect reuse of cache entries creates problems during remote caching. A
    bad cache entry in a shared cache affects every developer on the project,
    and wiping the entire remote cache is not a feasible solution.

-   Sandboxing mimics the behavior of remote execution — if a build works well
    with sandboxing, it will likely also work with remote execution. By making
    remote execution upload all necessary files (including local tools), you can
    significantly reduce maintenance costs for compile clusters compared to
    having to install the tools on every machine in the cluster every time you
    want to try out a new compiler or make a change to an existing tool.

## What sandbox strategy to use {:#sandboxing-strategies}

You can choose which kind of sandboxing to use, if any, with the
[strategy flags](user-manual.html#strategy-options). Using the `sandboxed`
strategy makes Bazel pick one of the sandbox implementations listed below,
preferring an OS-specific sandbox to the less hermetic generic one.
[Persistent workers](/remote/persistent) run in a generic sandbox if you pass
the `--worker_sandboxing` flag.

The `local` (a.k.a. `standalone`) strategy does not do any kind of sandboxing.
It simply executes the action's command line with the working directory set to
the execroot of your workspace.

`processwrapper-sandbox` is a sandboxing strategy that does not require any
"advanced" features - it should work on any POSIX system out of the box. It
builds a sandbox directory consisting of symlinks that point to the original
source files, executes the action's command line with the working directory set
to this directory instead of the execroot, then moves the known output artifacts
out of the sandbox into the execroot and deletes the sandbox. This prevents the
action from accidentally using any input files that are not declared and from
littering the execroot with unknown output files.

`linux-sandbox` goes one step further and builds on top of the
`processwrapper-sandbox`. Similar to what Docker does under the hood, it uses
Linux Namespaces (User, Mount, PID, Network and IPC namespaces) to isolate the
action from the host. That is, it makes the entire filesystem read-only except
for the sandbox directory, so the action cannot accidentally modify anything on
the host filesystem. This prevents situations like a buggy test accidentally rm
-rf'ing your $HOME directory. Optionally, you can also prevent the action from
accessing the network. `linux-sandbox` uses PID namespaces to prevent the action
from seeing any other processes and to reliably kill all processes (even daemons
spawned by the action) at the end.

`darwin-sandbox` is similar, but for macOS. It uses Apple's `sandbox-exec` tool
to achieve roughly the same as the Linux sandbox.

Both the `linux-sandbox` and the `darwin-sandbox` do not work in a "nested"
scenario due to restrictions in the mechanisms provided by the operating
systems. Because Docker also uses Linux namespaces for its container magic, you
cannot easily run `linux-sandbox` inside a Docker container, unless you use
`docker run --privileged`. On macOS, you cannot run `sandbox-exec` inside a
process that's already being sandboxed. Thus, in these cases, Bazel
automatically falls back to using `processwrapper-sandbox`.

If you would rather get a build error — such as to not accidentally build with a
less strict execution strategy — explicitly modify the list of execution
strategies that Bazel tries to use (for example, `bazel build
--spawn_strategy=worker,linux-sandbox`).

Dynamic execution usually requires sandboxing for local execution. To opt out,
pass the `--experimental_local_lockfree_output` flag. Dynamic execution silently
sandboxes [persistent workers](/remote/persistent).

## Downsides to sandboxing {:#sandboxing_downsides}

-   Sandboxing incurs extra setup and teardown cost. How big this cost is
    depends on many factors, including the shape of the build and the
    performance of the host OS. For Linux, sandboxed builds are rarely more than
    a few percent slower. Setting `--reuse_sandbox_directories` can
    mitigate the setup and teardown cost.

-   Sandboxing effectively disables any cache the tool may have. You can
    mitigate this by using [persistent workers](/remote/persistent), at
    the cost of weaker sandbox guarantees.

-   [Multiplex workers](/remote/multiplex) require explicit worker support
    to be sandboxed. Workers that do not support multiplex sandboxing run as
    singleplex workers under dynamic execution, which can cost extra memory.

## Debugging {:#debugging}

Follow the strategies below to debug issues with sandboxing.

### Deactivated namespaces {:#deactivated-namespaces}

On some platforms, such as
[Google Kubernetes Engine](https://cloud.google.com/kubernetes-engine/){: .external}
cluster nodes or Debian, user namespaces are deactivated by default due to
security concerns. If the `/proc/sys/kernel/unprivileged_userns_clone` file
exists and contains a 0, you can activate user namespaces by running:

```posix-terminal
   sudo sysctl kernel.unprivileged_userns_clone=1
```

### Rule execution failures {:#rule-failures}

The sandbox may fail to execute rules because of the system setup. If you see a
message like `namespace-sandbox.c:633: execvp(argv[0], argv): No such file or
directory`, try to deactivate the sandbox with `--strategy=Genrule=local` for
genrules, and `--spawn_strategy=local` for other rules.

### Detailed debugging for build failures {:#debugging-build-failures}

If your build failed, use `--verbose_failures` and `--sandbox_debug` to make
Bazel show the exact command it ran when your build failed, including the part
that sets up the sandbox.

Example error message:

```
ERROR: path/to/your/project/BUILD:1:1: compilation of rule
'//path/to/your/project:all' failed:

Sandboxed execution failed, which may be legitimate (such as a compiler error),
or due to missing dependencies. To enter the sandbox environment for easier
debugging, run the following command in parentheses. On command failure, a bash
shell running inside the sandbox will then automatically be spawned

namespace-sandbox failed: error executing command
  (cd /some/path && \
  exec env - \
    LANG=en_US \
    PATH=/some/path/bin:/bin:/usr/bin \
    PYTHONPATH=/usr/local/some/path \
  /some/path/namespace-sandbox @/sandbox/root/path/this-sandbox-name.params --
  /some/path/to/your/some-compiler --some-params some-target)
```

You can now inspect the generated sandbox directory and see which files Bazel
created and run the command again to see how it behaves.

Note that Bazel does not delete the sandbox directory when you use
`--sandbox_debug`. Unless you are actively debugging, you should disable
`--sandbox_debug` because it fills up your disk over time.

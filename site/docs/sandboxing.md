---
layout: documentation
title: Sandboxing
---

# Sandboxing

This article covers sandboxing in Bazel, installing `sandboxfs`, and debugging
your sandboxing environment.

_Sandboxing_ is a permission restricting strategy that isolates processes from
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
generated outputs. Bazel then uses an operating-system-provided
technique, containers on Linux and `sandbox-exec` on macOS, to constrain the
action within `execroot/`.

## Reasons for sandboxing

- Without action sandboxing, Bazel will not know if a tool uses undeclared input
  files (files that are not explicitly listed in the dependencies of an action).
  When one of the undeclared input files changes, Bazel still believes that the
  build is up-to-date and won’t rebuild the action-resulting in an incorrect
  incremental build.

- Incorrect reuse of cache entries creates problems during remote caching. A bad
  cache entry in a shared cache affects every developer on the project, and
  wiping the entire remote cache is not a feasible solution.

- Sandboxing is closely related to remote execution. If a build works well with
  sandboxing, it will likely work well with remote execution. Uploading all
  necessary files (including local tools) can significantly reduce maintenance
  costs for compile clusters compared to having to install the tools on every
  machine in the cluster every time you want to try out a new compiler or make
  a change to an existing tool.

## sandboxfs

`sandboxfs` is a FUSE file system that exposes an arbitrary view of the
underlying file system without time penalties. Bazel uses `sandboxfs` to
generate `execroot/` instantaneously for each action, avoiding the cost
of issuing thousands of system calls. Note that further I/O within `execroot/`
may be slower due to FUSE overhead.

### Install sandboxfs

Use the following steps to install `sandboxfs` and perform a Bazel build with
it:

**Download**

[Download and install](https://github.com/bazelbuild/sandboxfs/blob/master/INSTALL.md)
`sandboxfs` so that the `sandboxfs` binary ends up in your `PATH`.

**Run `sandboxfs`**

1. (macOS-only) [Install OSXFUSE](https://osxfuse.github.io/).
2. (macOS-only) Run
    ```shell
      sudo sysctl -w vfs.generic.osxfuse.tunables.allow_other=1
     ```
  You will need to do this after installation and after every reboot to ensure
  core macOS system services work through sandboxfs.
3. Run a Bazel build with `--experimental_use_sandboxfs`.
   ```shell
   $bazel build <target> --experimental_use_sandboxfs
   ```

**Troubleshooting**

If you see `local` instead of `darwin-sandbox` or `linux-sandbox` as an
annotation for the actions that are executed, this may mean that sandboxing is
disabled. Pass `--genrule_strategy=sandboxed --spawn_strategy=sandboxed` to
enable it.

## Debugging

Follow the strategies below to debug issues with sandboxing.

### Deactivated namespaces

On some platforms, such as [Google Kubernetes
Engine](https://cloud.google.com/kubernetes-engine/) cluster nodes or Debian,
user namespaces are deactivated by default due to security
concerns. If the `/proc/sys/kernel/unprivileged_userns_clone` file exists and
contains a 0, you can activate user namespaces by running:
```shell
   sudo sysctl kernel.unprivileged_userns_clone=1
   ```

### Rule execution failures

The sandbox may fail to execute rules because of the system setup.
If you see a message like `namespace-sandbox.c:633: execvp(argv[0], argv): No
such file or directory`, try to deactivate the sandbox with
`--strategy=Genrule=local` for genrules, and `--spawn_strategy=local`
for other rules.

### Detailed debugging for build failures

If your build failed, use `--verbose_failures` and `--sandbox_debug` to make
Bazel show the exact command it ran when your build failed, including the part
that sets up the sandbox.

Example error message:

```
ERROR: path/to/your/project/BUILD:1:1: compilation of rule
'//path/to/your/project:all' failed:

Sandboxed execution failed, which may be legitimate (e.g. a compiler error),
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

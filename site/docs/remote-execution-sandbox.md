---
layout: documentation
title: Troubleshooting Bazel Remote Execution with Docker Sandbox

---

# Troubleshooting Bazel Remote Execution with Docker Sandbox

## Contents

*  [Overview](#overview)
*  [Prerequisites](#prerequisites)
*  [Troubleshooting on the local machine](#troubleshooting-on-the-local-machine)
   *  [Step 1: Run the build](#step-1-run-the-build)
   *  [Step 2: Resolve detected issues](#step-2-resolve-detected-issues)
*  [Troubleshooting in a Docker container](#troubleshooting-in-a-docker-container)
   *  [Step 1: Build the container](#step-1-build-the-container)
   *  [Step 2: Start the container](#step-2-start-the-container)
   *  [Step 3: Test the container](#step-3-test-the-container)
   *  [Step 4: Run the build](#step-4-run-the-build)
   *  [Step 5: Resolve detected issues](#step-5-resolve-detected-issues)

## Overview

Bazel builds that succeed locally may fail when executed remotely due to
restrictions and requirements that do not affect local builds. The most common
causes of such failures are described in [Adapting Bazel Rules for Remote Execution](remote-execution-rules.html).

This document describes how to identify and resolve the most common issues that
arise with remote execution using the Docker sandbox feature, which imposes
restrictions upon the build equal to those of remote execution. This allows you
to troubleshoot your build without the need for a remote execution service.

The Docker sandbox feature mimics the restrictions of remote execution as
follows:

*   **Build actions execute in toolchain containers.** You can use the same
    toolchain containers to run your build locally and remotely via a service
    supporting containerized remote execution.

*   **No extraneous data crosses the container boundary.** Only explicitly
    declared inputs and outputs enter and leave the container, and only after
    the associated build action successfully completes.

*   **Each action executes in a fresh container.** A new, unique container is
    created for each spawned build action.

**Note:** Builds take noticeably more time to complete when the Docker sandbox
feature is enabled. This is normal.

You can troubleshoot these issues using one of the following methods:

*   **[Troubleshooting natively.](#troubleshooting-natively)** With this method,
    Bazel and its build actions run natively on your local machine. The Docker
    sandbox feature imposes restrictions upon the build equal to those of remote
    execution. However, this method will not detect local tools, states, and
    data leaking into your build, which will cause problems with remote execution.

*   **[Troubleshooting in a Docker container.](#troubleshooting-in-a-docker-container)**
    With this method, Bazel and its build actions run inside a Docker container,
    which allows you to detect tools, states, and data leaking from the local
    machine into the build in addition to imposing restrictions
    equal to those of remote execution. This method provides insight into your
    build even if portions of the build are failing. This method is experimental
    and not officially supported.

## Prerequisites

Before you begin troubleshooting, do the following if you have not already done so:

*   Install Docker and configure the permissions required to run it.
*   Install Bazel 0.14.1 or later. Earlier versions do not support the Docker
    sandbox feature.
*   Add the [bazel-toolchains](https://releases.bazel.build/bazel-toolchains.html)
    repo, pinned to the latest release version, to your build's `WORKSPACE` file
    as described [here](https://releases.bazel.build/bazel-toolchains.html).
*   Add flags to your `.bazelrc` file to enable the feature. Create the file in
    the root directory of your Bazel project if it does not exist. Flags below
    are a reference sample. Please see the latest
    [`.bazelrc`](https://github.com/bazelbuild/bazel-toolchains/tree/master/bazelrc)
    file in the bazel-toolchains repo and copy the values of the flags defined
    there for config `docker-sandbox`.

```
# Docker Sandbox Mode
build:docker-sandbox --host_javabase=<...>
build:docker-sandbox --javabase=<...>
build:docker-sandbox --crosstool_top=<...>
build:docker-sandbox --experimental_docker_image=<...>
build:docker-sandbox --spawn_strategy=docker --strategy=Javac=docker --genrule_strategy=docker
build:docker-sandbox --define=EXECUTOR=remote
build:docker-sandbox --experimental_docker_verbose
build:docker-sandbox --experimental_enable_docker_sandbox
```

**Note:** The flags referenced in the `.bazelrc` file shown above are configured
to run within the [`rbe-ubuntu16-04`](https://console.cloud.google.com/launcher/details/google/rbe-ubuntu16-04)
container.

If your rules require additional tools, do the following:

1.  Create a custom Docker container by installing tools using a [Dockerfile](https://docs.docker.com/engine/reference/builder/)
    and [building](https://docs.docker.com/engine/reference/commandline/build/)
    the image locally.

2.  Replace the value of the `--experimental_docker_image` flag above with the
    name of your custom container image.


## Troubleshooting natively

This method executes Bazel and all of its build actions directly on the local
machine and is a reliable way to confirm whether your build will succeed when
executed remotely.

However, with this method, locally installed tools, binaries, and data may leak
into into your build, especially if it uses [configure-style WORKSPACE rules](remote-execution-rules.html#managing-configure-style-workspace-rules).
Such leaks will cause problems with remote execution; to detect them, [troubleshoot in a Docker container](#troubleshooting-in-a-docker-container)
in addition to troubleshooting natively.

### Step 1: Run the build

1.  Add the `--config=docker-sandbox` flag to the Bazel command that executes
    your build. For example:

    ```
    bazel --bazelrc=.bazelrc build --config=docker-sandbox <target>

    ```

2.  Run the build and wait for it to complete. The build will run up to four
    times slower than normal due to the Docker sandbox feature.

You may encounter the following error:

```
ERROR: 'docker' is an invalid value for docker spawn strategy.
```

If you do, run the build again with the `--experimental_docker_verbose`  flag.
This flag enables verbose error messages. This error is typically caused by a
faulty Docker installation or lack of permissions to execute it under the
current user account. See the [Docker documentation](https://docs.docker.com/install/linux/linux-postinstall/)
for more information. If problems persist, skip ahead to [Troubleshooting in a Docker container](#troubleshooting-in-a-docker-container).

### Step 2: Resolve detected issues

The following are the most commonly encountered issues and their workarounds.

*  **A file, tool, binary, or resource referenced by the Bazel runfiles tree is
   missing.**. Confirm that all dependencies of the affected targets have been [explicitly declared](/build-ref.html#dependencies).
   See [Managing implicit dependencies](/remote-execution-rules.html#managing-implicit-dependencies)
   for more information.

*  **A file, tool, binary, or resource referenced by an absolute path or the `PATH`
   variable is missing.** Confirm that all required tools are installed within
   the toolchain container and use [toolchain rules](/toolchains.html) to properly
   declare dependencies pointing to the missing resource. See [Invoking build tools through toolchain rules](/remote-execution-rules.html#invoking-build-tools-through-toolchain-rules)
   for more information.

*  **A binary execution fails.** One of the build rules is referencing a binary
   incompatible with the execution environment (the Docker container). See [Managing platform-dependent binaries](/remote-execution-rules.html#managing-platform-dependent-binaries)
   for more information. If you cannot resolve the issue, contact [bazel-discuss@google.com](mailto:bazel-discuss@google.com)
   for help.

*  **A file from `@local-jdk` is missing or causing errors.** The Java binaries
   on your local machine are leaking into the build while being incompatible with
   it. Use [`java_toolchain`](be/java.html#java_toolchain)
   in your rules and targets instead of `@local_jdk`. Contact [bazel-discuss@google.com](mailto:bazel-discuss@google.com) if you need further help.

*  **Other errors.** Contact [bazel-discuss@google.com](mailto:bazel-discuss@google.com) for help.

## Troubleshooting in a Docker container

With this method, Bazel runs inside a host Docker container, and Bazel's build
actions execute inside individual toolchain containers spawned by the Docker
sandbox feature. The sandbox spawns a brand new toolchain container for each
build action and only one action executes in each toolchain container.

This method provides more granular control of tools installed in the host
environment. By separating the execution of the build from the execution of its
build actions and keeping the installed tooling to a minimum, you can verify
whether your build has any dependencies on the local execution environment.

### Step 1: Build the container

**Note:** The commands below are tailored specifically for a  `debian:stretch` base.
For other bases, modify them as necessary.

1.  Create a `Dockerfile` that creates the Docker container and installs Bazel
    with a minimal set of build tools:

    ```
    FROM debian:stretch

    RUN apt-get update && apt-get install -y apt-transport-https curl software-properties-common git gcc gnupg2 g++ openjdk-8-jdk-headless python-dev zip wget vim

    RUN curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add -

    RUN add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"

    RUN apt-get update && apt-get install -y docker-ce

    RUN wget https://releases.bazel.build/<latest Bazel version>/release/bazel-<latest Bazel version>-installer-linux-x86_64.sh -O ./bazel-installer.sh && chmod 755 ./bazel-installer.sh

    RUN ./bazel-installer.sh
    ```

2.  Build the container as `bazel_container`:

    ```
    docker build -t bazel_container - < Dockerfile
    ```

### Step 2: Start the container

Start the Docker container using the command shown below. In the command,
substitute the path to the source code on your host that you want to build.

```
docker run -it \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp:/tmp \
  -v <your source code directory>:/src \
  -w /src \
  bazel_container \
  /bin/bash
```

This command runs the container as root, mapping the docker socket, and mounting
the `/tmp` directory. This allows Bazel to spawn other Docker containers and to
use directories under `/tmp` to share files with those containers. Your source
code is available at `/src` inside the container.

The command intentionally starts from a `debian:stretch` base container that
includes binaries incompatible with the `rbe-ubuntu16-04` container used as a
toolchain container. If binaries from the local environment are leaking into the
toolchain container, they will cause build errors.

### Step 3: Test the container

Run the following commands from inside the Docker container to test it:

```
docker ps

bazel version
```

### Step 4: Run the build

Run the build as shown below. The output user is root so that it corresponds to
a directory that is accessible with the same absolute path from inside the host
container in which Bazel runs, from the toolchain containers spawned by the Docker
sandbox feature in which Bazel's build actions are running, and from the local
machine on which the host and action containers run.

```
bazel --output_user_root=/tmp/bazel_docker_root --bazelrc=.bazelrc \ build --config=docker-sandbox <target>
```

### Step 5: Resolve detected issues

You can resolve build failures as follows:

*   If the build fails with an "out of disk space" error, you  can increase this
    limit by starting the host container with the flag `--memory=XX` where `XX`
    is the allocated disk space in gigabytes. This is experimental and may
    result in unpredictable behavior.

*   If the build fails during the analysis or loading phases, one or more of
    your build rules declared in the WORKSPACE file are not compatible with
    remote execution. See [Adapting Bazel Rules for Remote Execution](remote-execution-rules.html)
    for possible causes and workarounds.

*   If the build fails for any other reason, see the troubleshooting steps in [Step 2: Resolve detected issues](#step-2-resolve-detected-issues).

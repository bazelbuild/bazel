Project: /_project.yaml
Book: /_book.yaml

# Getting Started with Bazel Docker Container

{% include "_buttons.html" %}

This page provides details on the contents of the Bazel container, how to build
the [abseil-cpp](https://github.com/abseil/abseil-cpp){: .external} project using Bazel
inside the Bazel container, and how to build this project directly
from the host machine using the Bazel container with directory mounting.

## Build Abseil project from your host machine with directory mounting {:#build-abseil}

The instructions in this section allow you to build using the Bazel container
with the sources checked out in your host environment. A container is started up
for each build command you execute. Build results are cached in your host
environment so they can be reused across builds.

Clone the project to a directory in your host machine.

```posix-terminal
git clone --depth 1 --branch 20220623.1 https://github.com/abseil/abseil-cpp.git /src/workspace
```

Create a folder that will have cached results to be shared across builds.

```posix-terminal
mkdir -p /tmp/build_output/
```

Use the Bazel container to build the project and make the build
outputs available in the output folder in your host machine.

```posix-terminal
docker run \
  -e USER="$(id -u)" \
  -u="$(id -u)" \
  -v /src/workspace:/src/workspace \
  -v /tmp/build_output:/tmp/build_output \
  -w /src/workspace \
  gcr.io/bazel-public/bazel:latest \
  --output_user_root=/tmp/build_output \
  build //absl/...
```

Build the project with sanitizers by adding the `--config={{ "<var>" }}asan{{ "</var>" }}|{{ "<var>" }}tsan{{ "</var>" }}|{{ "<var>" }}msan{{ "</var>" }}` build
flag to select AddressSanitizer (asan), ThreadSanitizer (tsan) or
MemorySanitizer (msan) accordingly.

```posix-terminal
docker run \
  -e USER="$(id -u)" \
  -u="$(id -u)" \
  -v /src/workspace:/src/workspace \
  -v /tmp/build_output:/tmp/build_output \
  -w /src/workspace \
  gcr.io/bazel-public/bazel:latest \
  --output_user_root=/tmp/build_output \
  build --config={asan | tsan | msan} -- //absl/... -//absl/types:variant_test
```

## Build Abseil project from inside the container {:#build-abseil-inside-container}

The instructions in this section allow you to build using the Bazel container
with the sources inside the container. By starting a container at the beginning
of your development workflow and doing changes in the worskpace within the
container, build results will be cached.

Start a shell in the Bazel container:

```posix-terminal
docker run --interactive --entrypoint=/bin/bash gcr.io/bazel-public/bazel:latest
```

Each container id is unique. In the instructions below, the container was 5a99103747c6.

Clone the project.

```posix-terminal
ubuntu@5a99103747c6:~$ git clone --depth 1 --branch 20220623.1 https://github.com/abseil/abseil-cpp.git && cd abseil-cpp/
```

Do a regular build.

```posix-terminal
ubuntu@5a99103747c6:~/abseil-cpp$ bazel build //absl/...
```

Build the project with sanitizers by adding the `--config={{ "<var>" }}asan{{ "</var>" }}|{{ "<var>" }}tsan{{ "</var>" }}|{{ "<var>" }}msan{{ "</var>" }}`
build flag to select AddressSanitizer (asan), ThreadSanitizer (tsan) or
MemorySanitizer (msan) accordingly.

```posix-terminal
ubuntu@5a99103747c6:~/abseil-cpp$ bazel build --config={asan | tsan | msan} -- //absl/... -//absl/types:variant_test
```

## Explore the Bazel container {:#explore-bazel-container}

If you haven't already, start an interactive shell inside the Bazel container.

```posix-terminal
docker run -it --entrypoint=/bin/bash gcr.io/bazel-public/bazel:latest
ubuntu@5a99103747c6:~$
```

Explore the container contents.

```posix-terminal
ubuntu@5a99103747c6:~$ gcc --version
gcc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

ubuntu@5a99103747c6:~$ java -version
openjdk version "1.8.0_362"
OpenJDK Runtime Environment (build 1.8.0_362-8u372-ga~us1-0ubuntu1~20.04-b09)
OpenJDK 64-Bit Server VM (build 25.362-b09, mixed mode)

ubuntu@5a99103747c6:~$ python -V
Python 3.8.10

ubuntu@5a99103747c6:~$ bazel version
WARNING: Invoking Bazel in batch mode since it is not invoked from within a workspace (below a directory having a WORKSPACE file).
Extracting Bazel installation...
Build label: 6.2.1
Build target: bazel-out/k8-opt/bin/src/main/java/com/google/devtools/build/lib/bazel/BazelServer_deploy.jar
Build time: Fri Jun 2 16:59:58 2023 (1685725198)
Build timestamp: 1685725198
Build timestamp as int: 1685725198
```

## Explore the Bazel Dockerfile {:#explore-bazel-dockerfile}

If you want to check how the Bazel Docker image is built, you can find its Dockerfile at [bazelbuild/continuous-integration/bazel/oci](https://github.com/bazelbuild/continuous-integration/tree/master/bazel/oci).

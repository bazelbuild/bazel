---
layout: documentation
title: Compile Bazel from Source
---

# <a name="compiling-from-source"></a>Compile Bazel from source

1. Ensure that you have OpenJDK 8 installed on your system.
   For a system based on debian packages (e.g. Debian, Ubuntu), install
   OpenJDK 8 by running the command `sudo apt-get install openjdk-8-jdk`.

2. The standard way of compiling a release version of Bazel from source is to
   use a distribution archive. Download `bazel-<VERSION>-dist.zip` from the
   [release page](https://github.com/bazelbuild/bazel/releases) for the desired
   version. We recommend to also verify the signature made by our
   [release key](https://bazel.build/bazel-release.pub.gpg) 48457EE0.
   The distribution archive also contains generated files in addition to the
   versioned sources, so this step _cannot_ be short cut by using a checkout
   of the source tree.

3. Unzip the archive and call `bash ./compile.sh`; this will create a bazel
   binary in `output/bazel`. This binary is self-contained, so it can be copied
   to a directory on the PATH (such as `/usr/local/bin`) or used in-place.

## <a name="compiling-from-source-issues"></a>Known issues when compiling from source

### On Windows:

* version 0.4.4 and below: `compile.sh` may fail right after start with an error
  like this:

    ```
    File not found - *.jar
    no error prone jar
    ```

    Workaround is to run this (and add it to your `~/.bashrc`):

    ```
    export PATH="/bin:/usr/bin:$PATH"
    ```

* version 0.4.3 and below: `compile.sh` may fail fairly early with many Java
  compilation errors. The errors look similar to:

    ```
    C:\...\bazel_VR1HFY7x\src\com\google\devtools\build\lib\remote\ExecuteServiceGrpc.java:11: error: package io.grpc.stub does not exist
    import static io.grpc.stub.ServerCalls.asyncUnaryCall;
                              ^
    ```

    This is caused by a bug in one of the bootstrap scripts
    (`scripts/bootstrap/compile.sh`). Manually apply this one-line fix if you
    want to build Bazel purely from source (without using an existing Bazel
    binary): [5402993a5e9065984a42eca2132ec56ca3aa456f]( https://github.com/bazelbuild/bazel/commit/5402993a5e9065984a42eca2132ec56ca3aa456f).

* version 0.3.2 and below:
  [github issue #1919](https://github.com/bazelbuild/bazel/issues/1919)

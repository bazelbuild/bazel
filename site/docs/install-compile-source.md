---
layout: documentation
title: Compile Bazel from Source
---

# <a name="compiling-from-source"></a>Compile Bazel from source

## <a name="unix"></a> On Linux or macOS

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

## <a name="windows"></a> On Windows

Windows support is in beta. Known issues are [marked with label
"Windows"](https://github.com/bazelbuild/bazel/issues?q=is%3Aissue+is%3Aopen+label%3A%22category%3A+multi-platform+%3E+windows%22)
on github issues.

Prerequisites:

*    Follow the [Windows requirements](windows.md#requirements).
*    Install [Visual C++ Build Tools](http://landinghub.visualstudio.com/visual-cpp-build-tools)
     or the full [Visual C++](https://www.visualstudio.com/) (as part of Visual
     Studio; Community Edition is fine) with Windows SDK installed.

To build Bazel on Windows:

*    Open the msys2 shell.
*    Clone the [Bazel git repository](https://github.com/bazelbuild/bazel) as normal.
*    Run ``compile.sh`` in Bazel directory.
*    If all works fine, Bazel will be built at ``output\bazel.exe``.

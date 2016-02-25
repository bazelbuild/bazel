---
layout: documentation
title: Windows
---

Building Bazel on Windows
=========================

Windows support is highly experimental. Known issues are [marked with
label "Windows"](https://github.com/bazelbuild/bazel/issues?q=is%3Aissue+is%3Aopen+label%3AWindows)
on github issues.

We currently support only 64 bit Windows 7 or higher and we can only compile
msys2 binaries.

To bootstrap Bazel on Windows, you will need:

*    Java JDK 8 or later
*    [msys2](https://msys2.github.io/) (need to be installed at
     ``C:\tools\msys64\``).
*    Several msys2 packages. Use the ``pacman`` command to install them:
     ``pacman -S gcc git curl zip unzip zlib-devel``

To build Bazel:

*    Open the msys2 shell.
*    Clone the Bazel git repository as normal.
*    Set the environment variables:

```bash
export JAVA_HOME="$(ls -d C:/Program\ Files/Java/jdk* | sort | tail -n 1)"
export TMPDIR=c:/temp
export BAZEL_SH=c:/tools/msys64/usr/bin/bash.exe
```

*     Run ``compile.sh`` in Bazel directory.
*     If all works fine, bazel will be built at ``output\bazel.exe``.

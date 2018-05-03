---
layout: documentation
title: Compiling Bazel from Source
---

# <a name="compiling-from-source"></a>Compiling Bazel from Source (bootstrapping)

You can build Bazel from source without using an existing Bazel binary.

### 1.  Install prerequisites

#### On Unix-like systems based on Debian packages (Debian, Ubuntu)

Ensure that JDK 8, Python, Bash, zip, and the usual C++ build toolchain are installed on your
system.

For example run following command in a terminal:

```sh
sudo apt-get install build-essential openjdk-8-jdk python zip
```

#### On Windows

Install:

*   [MSYS2 shell](https://msys2.github.io/)

*   Some MSYS2 packages:
    1.  Open the MSYS2 shell
    2.  Run `pacman -Syu zip unzip`

*   Visual C++ compiler

    Install the Visual C++ compiler either as part of Visual Studio 2015 or newer, or by installing
    the latest [Build Tools for Visual Studio 2017](https://aka.ms/BuildTools).

*   JDK 8

    Older and newer major versions are *not* supported.

*   Python 2 or Python 3

    Bazel needs the Windows-native Python (downloaded e.g. from https://www.python.org/), not the
    one installed by MSYS2 or pacman.

### 2.  Download and unpack Bazel's distribution archive

Download `bazel-<version>-dist.zip` from the
[release page](https://github.com/bazelbuild/bazel/releases).

**Tip:** We recommend to also verify the signature made by our
[release key](https://bazel.build/bazel-release.pub.gpg) 48457EE0.

**Note:** There are no architecture-specific or OS-specific distribution archives. This file is good
for all supported architectures and operating systems.

The distribution archive contains generated sources in addition to the versioned sources, therefore
you must use the distribution archive to build Bazel from scratch. You cannot use the source tree
you checked out from GitHub.

### 3.  Bootstrap Bazel

#### On Unix-like systems (e.g. Ubuntu, macOS)

1.  open a shell or Terminal window
2.  `cd` into the directory where you unpacked the distribution archive
3.  run `bash ./compile.sh`

The output will be `output/bazel`. This is a self-contained Bazel binary, without an embedded JDK.
You can copy it to a directory on the `PATH` (such as `/usr/local/bin` on Linux) or use it in-place.

#### On Windows

1.  open the MSYS2 shell
2.  set some environment variables:

    *   `BAZEL_VS` or `BAZEL_VC`: path to the Visual Studio directory, or to the Visaul C++
         directory, respectively. Setting one of them is enough.
    *   `JAVA_HOME`: the JDK directory
    *   `PATH`: must have the Python directory on it
    *   `BAZEL_SH`: path to bash.exe

    For example:
    ```sh
    export BAZEL_VS="C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools"
    export BAZEL_SH="C:/msys64/usr/bin/bash.exe"
    export PATH="/c/python27:$PATH"
    export JAVA_HOME="C:/Program Files/Java/jdk1.8.0_112"
    ```
    or
    ```sh
    export BAZEL_VC="C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC"
    export BAZEL_SH="c:/msys64/usr/bin/bash.exe"
    export PATH="/c/python27:$PATH"
    export JAVA_HOME="C:/Program Files/Java/jdk1.8.0_112"
    ```

3.  `cd` into the directory where you unpacked the distribution archive
4.  run `./compile.sh`

The output will be `output/bazel.exe`. This is a self-contained Bazel binary, without an embedded
JDK. You can copy it to a directory on the `%PATH%` or use it in-place.

You don't need to run Bazel from the MSYS2 shell. You can run Bazel from the Command Prompt
(`cmd.exe`) or PowerShell.

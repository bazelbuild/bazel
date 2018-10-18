---
layout: documentation
title: Compiling Bazel from Source
---

# <a name="compiling-from-source"></a>Compiling Bazel from Source (bootstrapping)

To build Bazel from source, you can do one of the following:

*   Build it using an existing Bazel Binary

*   Build it without an existing Bazel binary, which is known as _bootstraping_

This document describes how to bootstrap Bazel:

*   On [Ubuntu Linux, macOS, and other Unix-like systems](#unix-like)

*   On [Windows](#windows)

## Bootstrapping Bazel on Ubuntu Linux, macOS, and other Unix-like systems<a name="unix-like"></a>

### 1.  Install the prerequisites

Ensure you have installed:

*   **Bash**

*   **zip, unzip**

*   **C++ build toolchain**

*   **JDK 8.** You must install version 8 of the JDK. Versions other than 8 are
    *not* supported.

*   **Python**. Versions 2 and 3 are supported.

For example on Ubuntu Linux you can install these requirements using the following command:

```sh
sudo apt-get install build-essential openjdk-8-jdk python zip unzip
```

### 2.  Download and unpack Bazel's source files (distribution archive)<a name="download-dist"></a>

(This is the same for Windows.)

Download `bazel-<version>-dist.zip` from [GitHub](https://github.com/bazelbuild/bazel/releases)
(for example, `bazel-0.18.0-dist.zip`). If you want to build a development snapshot of
Bazel, download the archive for the latest release before the snapshot
you are interested in as you need to bootstrap that version of Bazel
first in order to build the snapshot version.

**Note:** There is a **single, architecture-independent** distribution archive. There are no
architecture-specific or OS-specific distribution archives.

**Note:** You have to use the distribution archive to build Bazel from source.
You cannot use a source tree cloned from GitHub. (The distribution archive
contains generated source files that are required for bootstrapping and are not
part of the normal Git source tree.)

We recommend to also verify the signature made by our
[release key](https://bazel.build/bazel-release.pub.gpg) 48457EE0.

### 3.  Bootstrap Bazel

1.  Open a shell or Terminal window.

2.  Change into the directory where you unpacked the distribution archive.

3.  Run the compilation script: `bash ./compile.sh`.

The compiled output is placed into `output/bazel`. This is a self-contained
Bazel binary, without an embedded JDK. You can copy it to a directory in the
`PATH` variable (such as `/usr/local/bin` on Linux) or use it in-place.

### 4. Development snapshots of Bazel<a name="dev-snapshot"></a>

(This is the same for Windows.)

Once you have a reasonably new working `bazel` binary, you can also build
snapshot versions of Bazel by checking out the commit you're interested in
and build a new version of Bazel by `bazel build //src:bazel`.

## Bootstrapping Bazel on Windows

### 1.  Install the prerequisites

Ensure you have installed:

*   [MSYS2 shell](https://msys2.github.io/)

*   **The required MSYS2 packages.** Run the following command in the MSYS2
    shell:

    ```
    pacman -Syu zip unzip
    ```

*   **The Visual C++ compiler.** Install the Visual C++ compiler either as part
    of Visual Studio 2015 or newer, or by installing the latest [Build Tools
    for Visual Studio 2017](https://aka.ms/BuildTools).

*   **JDK 8.** You must install version 8 of the JDK. Versions other than 8 are
    *not* supported.

*   **Python**. Versions 2 and 3 are supported. You need the
    Windows-native version (downloadable from [https://www.python.org](https://www.python.org)).
    Versions installed via pacman in MSYS2 will not work.

### 2.  Download and unpack Bazel's source files (distribution archive)

This step is the same as on Unix-like systems, [click here](#download-dist).

### 3.  Bootstrap Bazel

1.  Open the MSYS2 shell.

2.  Set the following environment variables:
    *   Either `BAZEL_VS` or `BAZEL_VC` (they are *not* the same): Set to the
        path to the Visual Studio directory (BAZEL\_V<b>S</b>) or to the Visual
        C++ directory (BAZEL\_V<b>C</b>). Setting one of them is enough.
    *   `BAZEL_SH`: Path of the MSYS2 `bash.exe`. See the command in the
        examples below.

        Do not set this to `C:\Windows\System32\bash.exe`. (You have that file
        if you installed Windows Subsystem for Linux.) Bazel does not support
        this version of `bash.exe`.
    *   `PATH`: Add the Python directory.
    *   `JAVA_HOME`: Set to the JDK directory.

    For example (using BAZEL\_V<b>S</b>):

        export BAZEL_VS="C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools"
        export BAZEL_SH="$(cygpath -m $(realpath $(which bash)))"
        export PATH="/c/python27:$PATH"
        export JAVA_HOME="C:/Program Files/Java/jdk1.8.0_112"

    or (using BAZEL\_V<b>C</b>):

        export BAZEL_VC="C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC"
        export BAZEL_SH="$(cygpath -m $(realpath $(which bash)))"
        export PATH="/c/python27:$PATH"
        export JAVA_HOME="C:/Program Files/Java/jdk1.8.0_112"

3.  Change into the directory where you unpacked the distribution archive.

4.  Run the compilation script: `./compile.sh`

The compiled output is placed into `output/bazel.exe`. This is a self-contained
Bazel binary, without an embedded JDK. You can copy it to a directory within the
`%PATH%` variable or use it in-place.

You don't need to run Bazel from the MSYS2 shell. You can run Bazel from the
Command Prompt (`cmd.exe`) or PowerShell.

### 4. Development snapshots of Bazel

This is the same as on Unix-like systems, [click here](#dev-snapshot).

---
layout: documentation
title: Compiling Bazel from source
---

<h1 id="compiling-from-source">Compiling Bazel from source</h1>

To build Bazel from source, you can do one of the following:

*   Build it [using an existing Bazel binary](#build-bazel-using-bazel)

*   Build it [without an existing Bazel binary](#bootstrap-bazel) which is known
    as _bootstraping_.

<h2 id="build-bazel-using-bazel">Build Bazel using Bazel</h2>

If you already have a Bazel binary, you can build Bazel from a GitHub checkout.

You will need:

*   A GitHub checkout of Bazel's sources at the desired commit.

*   The Bazel version that was the latest when the commit was merged. (Other
    Bazel versions may work too, but are not guaranteed to.) You can either
    download this version from
    [GitHub](https://github.com/bazelbuild/bazel/releases), or build it from
    source, or bootstrap it as described below.

*   The same prerequisites as for bootstrapping (JDK, C++ compiler, etc.)

Once you have a Bazel binary to build with and the source tree of Bazel, `cd`
into the directory and run `bazel build //src:bazel`.

<h2 id="bootstrap-bazel">Build Bazel from scratch (bootstrapping)</h2>

You can also build Bazel from scratch, without using an existing Bazel binary.

<h3 id="download-distfile">1. Download Bazel's sources (distribution archive)</h3>

(This step is the same for all platforms.)

1.  Download `bazel-<version>-dist.zip` from
    [GitHub](https://github.com/bazelbuild/bazel/releases), for example
    `bazel-0.18.0-dist.zip`.

    There is a **single, architecture-independent** distribution archive.  There
    are no architecture-specific or OS-specific distribution archives.

    You **have to use the distribution archive** to bootstrap Bazel. You cannot
    use a source tree cloned from GitHub. (The distribution archive contains
    generated source files that are required for bootstrapping and are not part
    of the normal Git source tree.)

2.  Unpack the zip file somewhere on disk.

    We recommend to also verify the signature made by our
    [release key](https://bazel.build/bazel-release.pub.gpg) 48457EE0.

**To build a development version** of Bazel from a GitHub checkout, you need a
working Bazel binary. [Scroll up](#build-bazel-using-bazel) to see how to build
Bazel using Bazel.

<h3 id="bootstrap-unix">2. Bootstrap Bazel on Ubuntu Linux, macOS, and other Unix-like systems</h3>

([Scroll down](#bootstrap-windows) for instructions for Windows.)

<h4 id="bootstrap-unix-prereq">2.1. Install the prerequisites</h4>

*   **Bash**

*   **zip, unzip**

*   **C++ build toolchain**

*   **JDK 8.** You must install version 8 of the JDK. Versions other than 8 are
    *not* supported.

*   **Python**. Versions 2 and 3 are supported, installing one of them is
    enough.

For example on Ubuntu Linux you can install these requirements using the
following command:

```sh
sudo apt-get install build-essential openjdk-8-jdk python zip unzip
```

<h4 id="bootstrap-unix-bootstrap">2.2. Bootstrap Bazel</h4>

1.  Open a shell or Terminal window.

3.  `cd` to the directory where you unpacked the distribution archive.

3.  Run the compilation script: `env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh`.

The compiled output is placed into `output/bazel`. This is a self-contained
Bazel binary, without an embedded JDK. You can copy it anywhere or use it
in-place. For convenience we recommend copying this binary to a directory that's
on your `PATH` (such as `/usr/local/bin` on Linux).

<h3 id="bootstrap-windows">2. Bootstrap Bazel on Windows</h3>

([Scroll up](#bootstrap-unix) for instructions for Linux, macOS, and other
Unix-like systems.)

<h4 id="bootstrap-windows-prereq">2.1. Install the prerequisites</h4>

*   [MSYS2 shell](https://msys2.github.io/)

*   **The MSYS2 packages for zip and unzip.** Run the following command in the MSYS2 shell:

    ```
    pacman -Syu zip unzip
    ```

*   **The Visual C++ compiler.** Install the Visual C++ compiler either as part
    of Visual Studio 2015 or newer, or by installing the latest [Build Tools
    for Visual Studio 2017](https://aka.ms/BuildTools).

*   **JDK 8.** You must install version 8 of the JDK. Versions other than 8 are
    *not* supported.

*   **Python**. Versions 2 and 3 are supported, installing one of them is
    enough. You need the Windows-native version (downloadable from
    [https://www.python.org](https://www.python.org)).  Versions installed via
    pacman in MSYS2 will not work.

<h4 id="bootstrap-windows-bootstrap">2.2. Bootstrap Bazel</h4>

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

    **Example** (using BAZEL\_V<b>S</b>):

        export BAZEL_VS="C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools"
        export BAZEL_SH="$(cygpath -m $(realpath $(which bash)))"
        export PATH="/c/python27:$PATH"
        export JAVA_HOME="C:/Program Files/Java/jdk1.8.0_112"

    or (using BAZEL\_V<b>C</b>):

        export BAZEL_VC="C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC"
        export BAZEL_SH="$(cygpath -m $(realpath $(which bash)))"
        export PATH="/c/python27:$PATH"
        export JAVA_HOME="C:/Program Files/Java/jdk1.8.0_112"

3.  `cd` to the directory where you unpacked the distribution archive.

4.  Run the compilation script: `./compile.sh`

The compiled output is placed into `output/bazel.exe`. This is a self-contained
Bazel binary, without an embedded JDK. You can copy it anywhere or use it
in-place. For convenience we recommend copying this binary to a directory that's
on your `PATH`.

You don't need to run Bazel from the MSYS2 shell. You can run Bazel from the
Command Prompt (`cmd.exe`) or PowerShell.

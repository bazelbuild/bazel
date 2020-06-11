---
layout: documentation
title: Compiling Bazel from source
---

<h1 id="compiling-from-source">Compiling Bazel from source</h1>

To build Bazel from source, you can do one of the following:

*   Build it [using an existing Bazel binary](#build-bazel-using-bazel)

*   Build it [without an existing Bazel binary](#bootstrap-bazel) which is known
    as _bootstrapping_.

<h2 id="build-bazel-using-bazel">Build Bazel using Bazel</h2>

TL;DR:

1.  Get the latest Bazel release from the
    [GitHub release page](https://github.com/bazelbuild/bazel/releases) or with
    [Bazelisk](https://github.com/bazelbuild/bazelisk).

2.  [Download Bazel's sources from GitHub](https://github.com/bazelbuild/bazel/archive/master.zip)
    and extract somewhere.
    Alternatively you can git clone the source tree from https://github.com/bazelbuild/bazel

3.  Install the same prerequisites as for bootstrapping (see
    [for Unix-like systems](#bootstrap-unix-prereq) or
    [for Windows](#bootstrap-windows-prereq))

4.  Build a development build of Bazel using Bazel:
    `bazel build //src:bazel-dev` (or `bazel build //src:bazel-dev.exe` on
    Windows)

5.  The resulting binary is at `bazel-bin/src/bazel-dev`
    (or `bazel-bin\src\bazel-dev.exe` on Windows). You can copy it wherever you
    like and use immediately without further installation.

Detailed instructions follow below.

<h3 id="build-bazel-install-bazel">Step 1: Get the latest Bazel release</h3>

**Goal**: Install or download a release version of Bazel. Make sure you can run
it by typing `bazel` in a terminal.

**Reason**: To build Bazel from a GitHub source tree, you need a pre-existing
Bazel binary. You can install one from a package manager or download one from
GitHub. See <a href="install.html">Installing Bazel</a>. (Or you can [build from
scratch (bootstrap)](#bootstrap-bazel).)

**Troubleshooting**:

*   If you cannot run Bazel by typing `bazel` in a terminal:

    *   Maybe your Bazel binary's directory is not on the PATH.

        This is not a big problem. Instead of typing `bazel`, you will need to
        type the full path.

    *   Maybe the Bazel binary itself is not called `bazel` (on Unixes) or
        `bazel.exe` (on Windows).

        This is not a big problem. You can either rename the binary, or type the
        binary's name instead of `bazel`.

    *   Maybe the binary is not executable (on Unixes).

        You must make the binary executable by running `chmod +x /path/to/bazel`.

<h3 id="build-bazel-git">Step 2: Download Bazel's sources from GitHub</h3>

If you are familiar with Git, then just git clone https://github.com/bazelbuild/bazel

Otherwise:

1.  Download the
    [latest sources as a zip file](https://github.com/bazelbuild/bazel/archive/master.zip).

2.  Extract the contents somewhere.

    For example create a `bazel-src` directory under your home directory and
    extract there.

<h3 id="build-bazel-prerequisites">Step 3: Install prerequisites</h3>

Install the same prerequisites as for bootstrapping (see below) -- JDK, C++
compiler, MSYS2 (if you are building on Windows), etc.

<h3 id="build-bazel-on-unixes">Step 4: Build Bazel</h3>

<h4 id="build-bazel-on-unixes">Ubuntu Linux, macOS, and other Unix-like systems</h4>

([Scroll down](#build-bazel-on-windows) for instructions for Windows.)

**Goal**: Run Bazel to build a custom Bazel binary (`bazel-bin/src/bazel-dev`).

**Instructions**:

1.  Start a Bash terminal

2.  `cd` into the directory where you extracted (or cloned) Bazel's sources.

    For example if you extracted the sources under your home directory, run:

        cd ~/bazel-src

3.  Build Bazel from source:

        bazel build //src:bazel-dev

    Alternatively you can run `bazel build //src:bazel --compilation_mode=opt`
    to yield a smaller binary but it's slower to build.

4.  The output will be at `bazel-bin/src/bazel-dev` (or `bazel-bin/src/bazel`).

<h4 id="build-bazel-on-windows">Windows</h4>

([Scroll up](#build-bazel-on-unixes) for instructions for Linux, macOS, and other
Unix-like systems.)

**Goal**: Run Bazel to build a custom Bazel binary
(`bazel-bin\src\bazel-dev.exe`).

**Instructions**:

1.  Start Command Prompt (Start Menu &gt; Run &gt; "cmd.exe")

2.  `cd` into the directory where you extracted (or cloned) Bazel's sources.

    For example if you extracted the sources under your home directory, run:

        cd %USERPROFILE%\bazel-src

3.  Build Bazel from source:

        bazel build //src:bazel-dev.exe

    Alternatively you can run `bazel build //src:bazel.exe
    --compilation_mode=opt` to yield a smaller binary but it's slower to build.

4.  The output will be at `bazel-bin\src\bazel-dev.exe` (or
    `bazel-bin\src\bazel.exe`).

<h3 id="build-bazel-install">Step 5: Install the built binary</h3>

Actually, there's nothing to install.

The output of the previous step is a self-contained Bazel binary. You can copy
it to any directory and use immediately. (It's useful if that directory is on
your PATH so that you can run "bazel" everywhere.)

---

<h2 id="bootstrap-bazel">Build Bazel from scratch (bootstrapping)</h2>

You can also build Bazel from scratch, without using an existing Bazel binary.

<h3 id="download-distfile">Step 1: Download Bazel's sources (distribution archive)</h3>

(This step is the same for all platforms.)

1.  Download `bazel-<version>-dist.zip` from
    [GitHub](https://github.com/bazelbuild/bazel/releases), for example
    `bazel-0.28.1-dist.zip`.

    **Attention**:

    -   There is a **single, architecture-independent** distribution archive.
        There are no architecture-specific or OS-specific distribution archives.
    -   These sources are **not the same as the GitHub source tree**. You
        have to use the distribution archive to bootstrap Bazel. You cannot
        use a source tree cloned from GitHub. (The distribution archive contains
        generated source files that are required for bootstrapping and are not part
        of the normal Git source tree.)

2.  Unpack the distribution archive somewhere on disk.

    We recommend to also verify the signature made by our
    [release key](https://bazel.build/bazel-release.pub.gpg) 3D5919B448457EE0.

<h3 id="bootstrap-unix">Step 2a: Bootstrap Bazel on Ubuntu Linux, macOS, and other Unix-like systems</h3>

([Scroll down](#bootstrap-windows) for instructions for Windows.)

<h4 id="bootstrap-unix-prereq">2.1. Install the prerequisites</h4>

*   **Bash**

*   **zip, unzip**

*   **C++ build toolchain**

*   **JDK.** Versions 8 and 11 are supported.

*   **Python**. Versions 2 and 3 are supported, installing one of them is
    enough.

For example on Ubuntu Linux you can install these requirements using the
following command:

```sh
sudo apt-get install build-essential openjdk-11-jdk python zip unzip
```

<h4 id="bootstrap-unix-bootstrap">2.2. Bootstrap Bazel</h4>

1.  Open a shell or Terminal window.

3.  `cd` to the directory where you unpacked the distribution archive.

3.  Run the compilation script: `env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh`.

The compiled output is placed into `output/bazel`. This is a self-contained
Bazel binary, without an embedded JDK. You can copy it anywhere or use it
in-place. For convenience we recommend copying this binary to a directory that's
on your `PATH` (such as `/usr/local/bin` on Linux).

To build the `bazel` binary in a reproducible way, also set
[`SOURCE_DATE_EPOCH`](https://reproducible-builds.org/specs/source-date-epoch/)
in the "Run the compilation script" step.

<h3 id="bootstrap-windows">Step 2b: Bootstrap Bazel on Windows</h3>

([Scroll up](#bootstrap-unix) for instructions for Linux, macOS, and other
Unix-like systems.)

<h4 id="bootstrap-windows-prereq">2.1. Install the prerequisites</h4>

*   [MSYS2 shell](https://msys2.github.io/)

*   **The MSYS2 packages for zip and unzip.** Run the following command in the MSYS2 shell:

    ```
    pacman -S zip unzip patch
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

4.  Run the compilation script: `env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" ./compile.sh`

The compiled output is placed into `output/bazel.exe`. This is a self-contained
Bazel binary, without an embedded JDK. You can copy it anywhere or use it
in-place. For convenience we recommend copying this binary to a directory that's
on your `PATH`.

To build the `bazel.exe` binary in a reproducible way, also set
[`SOURCE_DATE_EPOCH`](https://reproducible-builds.org/specs/source-date-epoch/)
in the "Run the compilation script" step.

You don't need to run Bazel from the MSYS2 shell. You can run Bazel from the
Command Prompt (`cmd.exe`) or PowerShell.

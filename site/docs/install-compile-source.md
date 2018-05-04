---
layout: documentation
title: Compiling Bazel from Source
---

# <a name="compiling-from-source"></a>Compiling Bazel from Source (bootstrapping)

You can build Bazel from source without using an existing Bazel binary.

### 1.  Install the prerequisites

#### Debian-based Unix systems

To compile Bazel on Debian-based systems such as Ubuntu or Debian, ensure that
JDK 8, Python, bash, zip, and the usual C++ build toolchain components are
installed on your system.

For example, you can install them using the following command:

```sh
sudo apt-get install build-essential openjdk-8-jdk python zip
```

#### Windows

To compile Bazel on Windows, install the following supporting software:

*   [MSYS2 shell](https://msys2.github.io/)

*   **The required MSYS2 packages.** Run the following command in the MSYS2
    shell:
    ```sh
    pacman -Syu zip unzip
    ```

*   **The Visual C++ compiler.** Install the Visual C++ compiler either as part
    of Visual Studio 2015 or newer, or by installing the latest [Build Tools
    for Visual Studio 2017](https://aka.ms/BuildTools).

*   **JDK 8.** You must install version 8 of the JDK. Versions other than 8 are
    *not* supported.

*   **Python**. Versions 2 and 3 are supported. You *must* install the
    Windows-native version (downloadable from https://www.python.org). Versions
    installed via pacman in MSYS2 will not work.

### 2.  Download and unpack Bazel's distribution archive.

Download `bazel-<version>-dist.zip` from the [release page](https://github.com/bazelbuild/bazel/releases).

**Note:** There is a **single, architecture-independent** distribution archive. There are no architecture-specific or OS-specific distribution archives.

We recommend to also verify the signature made by our [release key](https://bazel.build/bazel-release.pub.gpg) 48457EE0.

The distribution archive contains generated files in addition to the versioned sources, so this step _cannot_ be short cut by checking out the source tree.

### 3.  Bootstrap Bazel

#### Unix-like systems

On Unix-like systems such as Ubuntu or macOS, do the following:

1.  Open a shell or Terminal window.

2.  Change into the directory where you unpacked the distribution archive.

3.  Run the compilation script: `bash ./compile.sh`.

The compiled output is placed into `output/bazel`. This is a self-contained
Bazel binary, without an embedded JDK. You can copy it to a directory in the
`PATH` variable (such as `/usr/local/bin` on Linux) or use it in-place.

#### Windows

1.  Open the MSYS2 shell.

2.  Set the following environment variables:

    *   `BAZEL_VS` or `BAZEL_VC`: Set to the path to the Visual Studio directory
         or to the Visual C++ directory, respectively. Setting one of them is
         enough.

    *   `BAZEL_SH`: Set to the path of the MSYS2 `bash.exe`.

        Do not set this to `C:\Windows\System32\bash.exe`. (You have that file
        if you installed Windows Subsystem for Linux.) Bazel does not support
        this version of `bash.exe`.

    *   `PATH`: Add the Python directory.

    *   `JAVA_HOME`: Set to the JDK directory.

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

3.  Change into the directory where you unpacked the distribution archive.

4.  Run the compilation script: `./compile.sh`

The compiled output is placed into `output/bazel.exe`. This is a self-contained
Bazel binary, without an embedded JDK. You can copy it to a directory within the
`%PATH%` variable or use it in-place.

You don't need to run Bazel from the MSYS2 shell. You can run Bazel from the
Command Prompt (`cmd.exe`) or PowerShell.

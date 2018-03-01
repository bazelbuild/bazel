---
layout: documentation
title: Compiling Bazel from Source
---

# <a name="compiling-from-source"></a>Compiling Bazel from Source

You can build Bazel from source following these steps:

1.  Ensure that JDK 8, Python, bash, zip, and the usual C build toolchain
    are installed on your system.
    *   On systems based on Debian packages (Debian, Ubuntu): you can install
        OpenJDK 8 and Python by running the following command in a terminal:

        ```sh
        sudo apt-get install build-essential openjdk-8-jdk python zip
        ```
    *   On Windows: you need additional software. See the [requirements
        page](windows.html#requirements).

2.  Download and unpack Bazel's distribution archive.

    Download `bazel-<version>-dist.zip` from the [release
    page](https://github.com/bazelbuild/bazel/releases).

    **Note:** There is a **single architecture independent** distribution
    archive. There are no architecture-specific distribution archives.

    We recommend to also
    verify the signature made by our [release
    key](https://bazel.build/bazel-release.pub.gpg) 48457EE0.

    The distribution archive contains generated files in addition to the
    versioned sources, so this step _cannot_ be short cut by checking out the
    source tree.

3.  Build Bazel using `./compile.sh`.
    *   On Unix-like systems (e.g. Ubuntu, macOS), do the following steps in a
        shell session:
        1.  `cd` into the directory where you unpacked the distribution archive
        2.  run `bash ./compile.sh`
    *   On Windows, do following steps in the msys2 shell:
        1.  `cd` into the directory where you unpacked the distribution archive
        2.  run `./compile.sh`

    The output will be `output/bazel` on Unix-like systems (e.g. Ubuntu, macOS)
    and `output/bazel.exe` on Windows. This is a self-contained Bazel binary.
    You can copy it to a directory on the `PATH` (such as `/usr/local/bin` on
    Linux) or use it in-place.

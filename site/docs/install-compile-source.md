---
layout: documentation
title: Compile Bazel from Source
---

# <a name="compiling-from-source"></a>Compile Bazel from source

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
    page](https://github.com/bazelbuild/bazel/releases). We recommend to also
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

## Note to Windows users

Make sure your machine meets the [requirements](windows.html).

Depending on the Bazel version you compile:

*   `bazel-0.5.4-dist.zip` and newer:

    The compilation script uses Visual C++ to compile Bazel's C++ code. The
    resulting Bazel binary is a native Windows binary and it's not linked to
    `msys-2.0.dll`.

*   `bazel-0.5.0-dist.zip` through `bazel-0.5.3-dist.zip`:

    The compilation script uses GCC to build Bazel's C++ code. The resulting
    Bazel binary is an msys binary so it is linked to `msys-2.0.dll`.

    You can patch the compilation script to build Bazel using Visual C++ instead
    of GCC, and avoid linking it to `msys-2.0.dll`. The patch is to replace
    these lines in `./compile.sh`:

    ```sh
    if [[ $PLATFORM == "windows" ]]; then
      EXTRA_BAZEL_ARGS="${EXTRA_BAZEL_ARGS-} --cpu=x64_windows_msys --host_cpu=x64_windows_msys"
    fi
    ```

    with these lines:

    ```sh
    if [[ $PLATFORM == "windows" ]]; then
      EXTRA_BAZEL_ARGS="${EXTRA_BAZEL_ARGS-} --cpu=x64_windows_msvc --host_cpu=x64_windows_msvc"
    fi
    ```

*   `bazel-0.4.5-dist.zip` and earlier:

    The compilation script uses GCC to compile Bazel's C++ code. The resulting
    Bazel binary is an msys binary so it is linked to `msys-2.0.dll`. Bazel
    cannot be built using Visual C++.

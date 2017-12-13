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

## Note to Windows users

Make sure your machine meets the [requirements](windows.html) and that you use
the latest version of the sources (`bazel-0.X.Y-dist.zip`).

There's a bug in the compilation scripts in `bazel-0.6.0-dist.zip` and in
`bazel-0.6.1-dist.zip`:

To fix it:

*   either apply the changes in
    [e79a110](https://github.com/bazelbuild/bazel/commit/e79a1107d90380501102990d82cbfaa8f51a1778)
    to the source tree,

*   or just replace the following line in
    `src/main/native/windows/build_windows_jni.sh`:

     ```sh
     @CL /O2 /EHsc /LD /Fe:"$(cygpath -a -w ${DLL})" /I "${VSTEMP}" /I . ${WINDOWS_SOURCES[*]}
     ```

     with this line:

     ```sh
     @CL /O2 /EHsc /LD /Fe:"$(cygpath -a -w ${DLL})" /I "%TMP%" /I . ${WINDOWS_SOURCES[*]}
     ```

It suffices to do one of these to bootstrap Bazel. We however recommend
applying the full commit (e79a110) because it also adds extra environment
checks to `./compile.sh`.

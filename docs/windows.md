---
layout: documentation
---

Building Bazel on Windows
=========================

Warning: Windows support on Bazel is still at a very early stage, many things
will not work.

Since Bazel is written in Java, you will need a recent
[JDK](http://www.oracle.com/technetwork/java/javase/downloads/index.html).

The Windows build process uses MSYS2, a POSIX-like environment on Windows. Grab
the [installer](http://sourceforge.net/projects/msys2/files/Base/x86_64/)
or an archived version and install.

Next, open an MSYS2 shell (not the mingw32 or the mingw64 shell) either through
a shortcut or by running the `msys2_shell.bat`. Install the dependencies:

    pacman -S libarchive-devel gcc mingw-w64-x86_64-gcc zip unzip git

The msys2 gcc will be used for building Bazel itself, the mingw-w64 gcc will
be used by Bazel for building C++.

Grab the Bazel source code:

    git clone https://github.com/google/bazel.git

Then, to build:

    cd bazel && ./compile.sh

If all goes well, you should find the binary in `output/bazel`.


Running Bazel on Windows
========================

Running Bazel on Windows requires a few additional steps due to the differences
between the native POSIX environment and the Windows environment.

First, since Blaze uses symlinks, the current Windows user needs to have the
permission to create symlinks. This permission is off by default for
non-administrator users. The easiest workaround is to run the msys2 shell
as administrator (right click shortcut -> Run As Administrator).

Second, you need to set some environment variables:

    export MSYS=winsymlinks:nativestrict  # Enable symlink support in msys2.
    export BAZEL_SH="C:/msys64/usr/bin/bash"
    export JAVA_HOME="$(ls -d C:/Program\ Files/Java/jdk* | sort | tail -n 1)"

Third, you need to set some Bazel options. It's easiest to put them into your
`.bazelrc`:

    cat > ~/.bazelrc << EOF
    startup --batch  # Server mode is not yet supported on Windows
    build --compiler=windows_msys64_mingw64
    build --nobuild_runfile_links  # Not ported to Windows yet.
    EOF

This should be enough to run Bazel:

    ./output/bazel build //examples/cpp:hello-world

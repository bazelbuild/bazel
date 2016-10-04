---
layout: documentation
title: Windows
---

Building Bazel on Windows
=========================

Windows support is highly experimental. Known issues are [marked with
label "Windows"](https://github.com/bazelbuild/bazel/issues?q=is%3Aissue+is%3Aopen+label%3A%22category%3A+multi-platform+%3E+windows%22)
on github issues.

We currently support only 64 bit Windows 7 or higher and we compile Bazel as a msys2 binary.

To bootstrap Bazel on Windows, you will need:

*    Java JDK 8 or later
*    [Visual Studio](https://www.visualstudio.com/) (Community Edition is okay)
*    [msys2](https://msys2.github.io/) (need to be installed at
     ``C:\tools\msys64\``).
*    Several msys2 packages. Use the ``pacman`` command to install them:
     ``pacman -Syuu gcc git curl zip unzip zlib-devel``

To build Bazel:

*    Open the msys2 shell.
*    Clone the Bazel git repository as normal.
*    Set the environment variables:

```bash
export JAVA_HOME="$(ls -d C:/Program\ Files/Java/jdk* | sort | tail -n 1)"
export BAZEL_SH=c:/tools/msys64/usr/bin/bash.exe
```

*     Run ``compile.sh`` in Bazel directory.
*     If all works fine, bazel will be built at ``output\bazel.exe``.

Using Bazel on Windows
======================

Bazel now supports building C++, Java and Python targets on Windows.

### Build C++

To build C++ targets, you will need:

* [Visual Studio](https://www.visualstudio.com/)
<br/>We are using MSVC as the native C++ toolchain, so please ensure you have Visual
Studio installed with the Visual C++ components
(which is NOT the default installation type of Visual Studio).
You can set BAZEL\_VS environment variable to tell Bazel
where Visual Studio is, otherwise Bazel will try to find the latest version installed.
<br/>For example: `export BAZEL_VS="C:/Program Files (x86)/Microsoft Visual Studio 14.0"`

* [Python 2.7](https://www.python.org/downloads/)
<br/>Currently, we use python wrapper scripts to call the actual MSVC compiler, so
please make sure Python is installed and its location is added into PATH.
It's also a good idea to set BAZEL\_PYTHON environment variable to tell Bazel
where python is.
<br/>For example: `export BAZEL_PYTHON=C:/Python27/python.exe`

Bazel will auto-configure the location of Visual Studio and Python at the first
time you build any target.
If you need to auto-configure again, just run `bazel clean` then build a target.

If everything is set up, you can build C++ target now! However, since MSVC
toolchain is not default on Windows yet, you should use flag
`--cpu=x64_windows_msvc` to enable it like this:

```bash
$ bazel build --cpu=x64_windows_msvc examples/cpp:hello-world
$ ./bazel-bin/examples/cpp/hello-world.exe
$ bazel run --cpu=x64_windows_msvc examples/cpp:hello-world
```

### Build Java

Building Java targets works well on Windows, no special configuration is needed.
Just try:

```bash
$ bazel build examples/java-native/src/main/java/com/example/myproject:hello-world
$ ./bazel-bin/examples/java-native/src/main/java/com/example/myproject/hello-world
$ bazel run examples/java-native/src/main/java/com/example/myproject:hello-world
```

### Build Python

On Windows, we build a self-extracting zip file for executable python targets, you can even use
`python ./bazel-bin/path/to/target` to run it in native Windows command line (cmd.exe).
See more details in this [design doc](/designs/2016/09/05/build-python-on-windows.html).

```bash
$ bazel build examples/py_native:bin
$ ./bazel-bin/examples/py_native/bin
$ python ./bazel-bin/examples/py_native/bin    # This works in both msys and cmd.exe
$ bazel run examples/py_native:bin
```

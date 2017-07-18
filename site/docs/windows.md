---
layout: documentation
title: Windows
---

# Using Bazel on Windows

Windows support is in beta. Known issues are [marked with label
"Windows"](https://github.com/bazelbuild/bazel/issues?q=is%3Aissue+is%3Aopen+label%3A%22category%3A+multi-platform+%3E+windows%22)
on github issues.

We currently support only 64 bit Windows 7 or higher and we compile Bazel as a
msys2 binary.

## <a name="requirements"></a>Requirements

Before you can compile or run Bazel, you will need to set some environment
variables:

```bash
export JAVA_HOME="$(ls -d C:/Program\ Files/Java/jdk* | sort | tail -n 1)"
export BAZEL_SH=c:/tools/msys64/usr/bin/bash.exe
```

If you run outside of `bash`, ensure that ``msys-2.0.dll`` is in your ``PATH``
(if you install msys2 to ``c:\tools\msys64``, just add
``c:\tools\msys64\usr\bin`` to ``PATH``).

If you have another tool that vendors msys2 (such as msysgit), then
``c:\tools\msys64\usr\bin`` must appear in your ``PATH`` *before* entries for
those tools.

Similarly, if you have [bash on Ubuntu on
Windows](https://msdn.microsoft.com/en-gb/commandline/wsl/about) installed, you
should make sure ``c:\tools\msys64\usr\bin`` appears in ``PATH`` *before*
``c:\windows\system32``, because otherwise Windows' ``bash.exe`` is used before
msys2's.

Use ``where msys-2.0.dll`` to ensure your ``PATH`` is set up correctly.

To **run** Bazel (even pre-built binaries), you will need:

*    Java JDK 8 or later
*    [msys2 shell](https://msys2.github.io/) (need to be installed at
     ``C:\tools\msys64\``).
    * We build against version
     [20160205](https://sourceforge.net/projects/msys2/files/Base/x86_64/msys2-x86_64-20160205.exe/download),
     you will need this version in order to run the pre-built
     [release Bazel binaries](https://github.com/bazelbuild/bazel/releases).
    * You can also use newer versions or the
     [latest version](https://sourceforge.net/projects/msys2/files/latest/download?source=files),
     but then you will need to compile Bazel from the distribution archive (the
     source zip file) so that it's linked against the right version of
     ``msys-2.0.dll``. See also the
     [known issues](install-compile-source.md#known-issues-when-compiling-from-source).
*    Several msys2 packages. Use the ``pacman`` command to install them:

     ```
     pacman -Syuu gcc git curl zip unzip zlib-devel
     ```

## <a name="install"></a>Installation

See [Install Bazel on Windows](install-windows.md) for installation instructions.

## <a name="using"></a>Using Bazel on Windows

Bazel now supports building C++, Java and Python targets on Windows.

### Build C++

To build C++ targets, you will need:

* [Visual Studio](https://www.visualstudio.com/)
<br/>We are using MSVC as the native C++ toolchain, so please ensure you have Visual
Studio installed with the `Visual C++ > Common Tools for Visual C++` and
`Visual C++ > Microsoft Foundation Classes for C++` features.
(which is NOT the default installation type of Visual Studio).
You can set `BAZEL_VS` environment variable to tell Bazel
where Visual Studio is, otherwise Bazel will try to find the latest version installed.
<br/>For example: `export BAZEL_VS="C:/Program Files (x86)/Microsoft Visual Studio 14.0"`

* [Python](https://www.python.org/downloads/)
<br/>Both Python 2 and Python 3 are supported.
Currently, we use Python wrapper scripts to call the actual MSVC compiler, so
please make sure Python is installed and its location is added into PATH.
It's also a good idea to set `BAZEL_PYTHON` environment variable to tell Bazel
where Python is.
<br/>For example: `export BAZEL_PYTHON=C:/Python27/python.exe`

Bazel will auto-configure the location of Visual Studio and Python at the first
time you build any target.
If you need to auto-configure again, just run `bazel clean` then build a target.

If everything is set up, you can build C++ target now!

```bash
bazel build examples/cpp:hello-world
./bazel-bin/examples/cpp/hello-world.exe
bazel run examples/cpp:hello-world
```

However, with Bazel version prior to 0.5.0, MSVC
toolchain is not default on Windows, you should use flag
`--cpu=x64_windows_msvc` to enable it like this:

```bash
bazel build --cpu=x64_windows_msvc examples/cpp:hello-world
```

### Build Java

Building Java targets works well on Windows, no special configuration is needed.
Just try:

```bash
bazel build examples/java-native/src/main/java/com/example/myproject:hello-world
./bazel-bin/examples/java-native/src/main/java/com/example/myproject/hello-world
bazel run examples/java-native/src/main/java/com/example/myproject:hello-world
```

### Build Python

On Windows, we build a self-extracting zip file for executable Python targets, you can even use
`python ./bazel-bin/path/to/target` to run it in native Windows command line (cmd.exe).
See more details in this [design doc](https://bazel.build/designs/2016/09/05/build-python-on-windows.html).

```bash
bazel build examples/py_native:bin
./bazel-bin/examples/py_native/bin
python ./bazel-bin/examples/py_native/bin    # This works in both msys and cmd.exe
bazel run examples/py_native:bin
```

---
layout: documentation
title: Windows
---

# Using Bazel on Windows

## Windows version requirements

Bazel is a native Windows binary.

Bazel runs on 64 bit Windows 7 or higher, and on equivalent Windows Server
versions. Check
<a href="https://msdn.microsoft.com/en-us/library/windows/desktop/ms724832(v=vs.85).aspx">Microsoft's
Operating System Version table</a> to see if your OS is supported.

## Known issues

We mark Windows-related Bazel issues on GitHub with the "multi-platform >
windows" label. [You can see the open issues here.](https://github.com/bazelbuild/bazel/issues?q=is%3Aissue+is%3Aopen+label%3A%22category%3A+multi-platform+%3E+windows%22)

## Running Bazel: MSYS2 shell vs. Command Prompt vs. PowerShell

It's best to run Bazel from the Command Prompt (`cmd.exe`) or from PowerShell.

You can also run Bazel from the MSYS2 shell, but you need to disable MSYS2's
automatic path conversion. See [this StackOverflow
answer](https://stackoverflow.com/a/49004265/7778502) for details.

## <a name="requirements"></a>Software requirements

*   Python 2.7 or later.

    Use the Windows-native Python version. Do not use Python that comes with the
    MSYS2 shell or that you installed in MSYS using Pacman because it doesn't
    work with Bazel.

*   [MSYS2 shell](https://msys2.github.io/).

    You also need to set the `BAZEL_SH` environment variable to point to
    `bash.exe`. For example in the Windows Command Prompt (`cmd.exe`):

    ```
    set BAZEL_SH=C:\tools\msys64\usr\bin\bash.exe
    ```

    **Note**: do not use quotes (") around the path like you would on Unixes.
    Windows doesn't need them and it may confuse Bazel.

*   Several MSYS2 packages.

    Run the following command in the MSYS2 shell to install them:

    ```bash
    pacman -Syuu git curl zip unzip
    ```

*   Java JDK 8.

    JDK 7 and 9 are not supported.

    This step is not required if you downloaded a binary distribution of Bazel
    because it has JDK 8 embedded.

*   If you built Bazel from source: set the `JAVA_HOME` environment variable to
    the JDK's directory.

    For example in the Windows Command Prompt (`cmd.exe`):

    ```
    set JAVA_HOME=C:\Program Files\Java\jdk1.8.0_112
    ```

    **Note**: do not use quotes (") around the path like you would on Unix.
    Windows doesn't need them and they may confuse Bazel.

    This step is not required if you downloaded a binary distribution of Bazel
    or installed Bazel using Chocolatey. See [installing Bazel on
    Windows](install-windows.html).

*   [Microsoft Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145)

    This may already be installed on your system.


## Setting environment variables

Environment variables you set in the Windows Command Prompt (`cmd.exe`) are only
set in that command prompt session. If you start a new `cmd.exe`, you need to
set the variables again. To always set the variables when `cmd.exe` starts, you
can add them to the User variables or System variables in the `Control Panel >
System Properties > Advanced > Environment Variables...` dialog box.

## <a name="install"></a>Installation

See [Install Bazel on Windows](install-windows.html) for installation
instructions.

## <a name="using"></a>Using Bazel on Windows

The first time you build any target, Bazel auto-configures the location of
Python and the Visual C++ compiler. If you need to auto-configure again, run
`bazel clean` then build a target.

You can also tell Bazel where to find the Python binary and the C++ compiler:
- use the [`--python_path=c:\path\to\python.exe`](command-line-reference.html#flag--python_path) flag for Python
- use the `BAZEL_VC` or `BAZEL_VS` environment variable. See the [Build
  C++ section](#build_cpp) below.

### <a name="build_cpp"></a>Build C++

To build C++ targets, you need:

*   The Visual C++ compiler.

    You can install it in one of the following ways:

    *   Install [Visual Studio 2015 or later](https://www.visualstudio.com/)
        (Community Edition is enough) with Visual C++.

        Make sure to also install the `Visual C++ > Common Tools for Visual C++`
        and `Visual C++ > Microsoft Foundation Classes for C++` features. These
        features are not installed by default.

    *   Install the [Visual C++ Build
        Tools 2015 or later](http://landinghub.visualstudio.com/visual-cpp-build-tools).

        Due to a [known issue](https://github.com/bazelbuild/bazel/issues/3949) with
        Bazel's support for the Build Tools for Visual Studio 2017, we currently
        recommend using the 2015 version instead.

*   The `BAZEL_VS` or `BAZEL_VC` environment variable.

    Bazel tries to locate the C++ compiler the first time you build any
    target. To tell Bazel where the compiler is, you can set one of the
    following environment variables:

    *   `BAZEL_VS` storing the Visual Studio installation directory

    *   `BAZEL_VC` storing the Visual C++ Build Tools installation directory

    Setting one of these variables is enough. For example:

    ```
    set BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio 14.0
    ```

    or

    ```
    set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC
    ```

*   The [Windows
    SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk).

    The Windows SDK contains header files and libraries you need when building
    Windows applications, including Bazel itself.

If everything is set up, you can build a C++ target now!

Try building a target from one of our [sample
projects](https://github.com/bazelbuild/bazel/tree/master/examples):

```
C:\projects\bazel> bazel build //examples/cpp:hello-world

C:\projects\bazel> bazel-bin\examples\cpp\hello-world.exe
```

### Build Java

There's no setup necessary.

On Windows, Bazel builds two output files for `java_binary` rules:

*   a `.jar` file
*   a `.exe` file that can set up the environment for the JVM and run the binary

Try building a target from one of our [sample
projects](https://github.com/bazelbuild/bazel/tree/master/examples):

```
C:\projects\bazel> bazel build //examples/java-native/src/main/java/com/example/myproject:hello-world

C:\projects\bazel> bazel-bin\examples\java-native\src\main\java\com\example\myproject\hello-world.exe
```

### Build Python

To build Python targets, you need:

*   The [Python interpreter](https://www.python.org/downloads/)

    Both Python 2 and Python 3 are supported.

    To tell Bazel where Python is, you can use `--python_path=<path/to/python>`.
    For example:

    ```
    bazel build --python_path=C:/Python27/python.exe ...
    ```

    If `--python_path` is not specified, Bazel uses `python.exe` as
    the interpreter and the binary looks for it in `$PATH` during runtime.
    If it is not in `$PATH`(for example, when you use `py_binary` as an action's
    executable, Bazel will sanitize `$PATH`), then the execution will fail.


On Windows, Bazel builds two output files for `py_binary` rules:

*   a self-extracting zip file
*   an executable file that can launch the Python interpreter with the
    self-extracting zip file as the argument

You can either run the executable file (it has a `.exe` extension) or you can run
Python with the self-extracting zip file as the argument.

Try building a target from one of our [sample
projects](https://github.com/bazelbuild/bazel/tree/master/examples):

```
C:\projects\bazel> bazel build //examples/py_native:bin

C:\projects\bazel> bazel-bin\examples\py_native\bin.exe

C:\projects\bazel> python bazel-bin\examples\py_native\bin.zip
```

If you are interested in details about how Bazel builds Python targets on
Windows, check out this [design
doc](https://bazel.build/designs/2016/09/05/build-python-on-windows.html).

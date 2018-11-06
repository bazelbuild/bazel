---
layout: documentation
title: Windows
---

# Using Bazel on Windows

## <a name="install"></a>Installation

See [Install Bazel on Windows](install-windows.html) for installation
instructions.

## Known issues

We mark Windows-related Bazel issues on GitHub with the "platform:windows"
label. [You can see the open issues here.](https://github.com/bazelbuild/bazel/issues?q=is%3Aopen+is%3Aissue+label%3A%22platform%3A+windows%22)

## Running Bazel: MSYS2 shell vs. Command Prompt vs. PowerShell

It's best to run Bazel from the Command Prompt (`cmd.exe`) or from PowerShell.

You can also run Bazel from the MSYS2 shell, but you need to disable MSYS2's
automatic path conversion. See [this StackOverflow
answer](https://stackoverflow.com/a/49004265/7778502) for details.

## Setting environment variables

Environment variables you set in the Windows Command Prompt (`cmd.exe`) are only
set in that command prompt session. If you start a new `cmd.exe`, you need to
set the variables again. To always set the variables when `cmd.exe` starts, you
can add them to the User variables or System variables in the `Control Panel >
System Properties > Advanced > Environment Variables...` dialog box.

## <a name="using"></a>Using Bazel on Windows

The first time you build any target, Bazel auto-configures the location of
Python and the Visual C++ compiler. If you need to auto-configure again, run
`bazel clean` then build a target.

You can also tell Bazel where to find the Python binary and the C++ compiler:
- use the [`--python_path=c:\path\to\python.exe`](command-line-reference.html#flag--python_path) flag for Python
- use the `BAZEL_VC` or the `BAZEL_VS` environment variable (they are *not* the same!).
  See the [Build C++ section](#build_cpp) below.

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
        Tools 2015 or later](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017).

        If [alwayslink](be/c-cpp.html#cc_library.alwayslink) doesn't work with
        VS 2017, that is due to a
        [known issue](https://github.com/bazelbuild/bazel/issues/3949),
        please upgrade your VS 2017 to the latest version.

*   The `BAZEL_VS` or `BAZEL_VC` environment variable. (They are *not* the same!)

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

    The first command sets the path to Visual Studio (BAZEL\_V<b>S</b>), the other
    sets the path to Visual C++ (BAZEL\_V<b>C</b>).
    
    For Visual Studio 2017, with a default install, instead you might want
    
    ```
    set BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools
    ```
    
    or
    
     ```
    set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC
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

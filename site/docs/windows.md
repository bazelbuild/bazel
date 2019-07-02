---
layout: documentation
title: Windows
---

# Using Bazel on Windows

<a name="install"></a>
## Installation

See [Install Bazel on Windows](install-windows.html) for installation
instructions.

## Known issues

We mark Windows-related Bazel issues on GitHub with the "team-Windows"
label. [You can see the open issues here.](https://github.com/bazelbuild/bazel/issues?q=is%3Aopen+is%3Aissue+label%3Ateam-Windows)

<a name="running-bazel-shells"></a>
## Running Bazel: MSYS2 shell vs. Command Prompt vs. PowerShell

It's best to run Bazel from the Command Prompt (`cmd.exe`) or from PowerShell.

You can also run Bazel from the MSYS2 shell, but you need to disable MSYS2's
automatic path conversion. See [this StackOverflow
answer](https://stackoverflow.com/a/49004265/7778502) for details.

<a name="using-bazel-without-bash"></a>
## Using Bazel without Bash (MSYS2)

<a name="bazel-build-without-bash"></a>
### `bazel build` without Bash

With **Bazel 0.26.0** and the `--incompatible_windows_native_test_wrapper` flag,
you can **build Python and all C++ rules without Bash**. Use the
`--shell_executable=""` flag to tell Bazel not to look for Bash.

With **Bazel 0.25.0** and the `--incompatible_windows_native_test_wrapper` flag,
you can **build Java and `cc_binary` rules without Bash** (but not `cc_test`).
Use the `--shell_executable=""` flag to tell Bazel not to look for Bash.

With **Bazel 0.24.x and older** you need Bash to build any rule.

With every Bazel version, you **still need Bash** if a rule in your build or in
some external repository:

- is a `genrule`, because genrules execute Bash commands
- is a `sh_binary` or `sh_test` rule, because these inherently need Bash
- is a Starlark rule that uses `ctx.actions.run_shell()` or
  `ctx.resolve_command()`

However, `genrule` is often used for simple tasks like
[copying a file](https://github.com/bazelbuild/bazel-skylib/blob/master/rules/copy_file.bzl)
or [writing a text file](https://github.com/bazelbuild/bazel-skylib/blob/master/rules/write_file.bzl).
Instead of using `genrule` (and depending on Bash) you may find a suitable rule
in the
[bazel-skylib repository](https://github.com/bazelbuild/bazel-skylib/tree/master/rules).
When built on Windows, **these rules do not require Bash**.

<a name="bazel-test-without-bash"></a>
### `bazel test` without Bash

With **Bazel 0.25.0 or newer** and the
`--incompatible_windows_native_test_wrapper` flag, you can `bazel test` rules
without Bash, i.e.
`bazel test --incompatible_windows_native_test_wrapper //foo:bar_test` works
even if there's no MSYS2 installed.

With **Bazel 0.24.x and older** you cannot use this flag, and need Bash (MSYS2)
to run any `bazel test`.

In Bazel 0.25.0 and Bazel 0.26.0, the
`--incompatible_windows_native_test_wrapper` flag is **off** be default. We plan
to enable it by default starting with Bazel 0.27.0, and plan to remove support
for the flag in Bazel 0.28.0. Follow issue
[#6622](https://github.com/bazelbuild/bazel/pull/6622) for updates.

<a name="bazel-run-without-bash"></a>
### `bazel run` without Bash

With Bazel 0.25.0 you still need Bash (MSYS2) to `bazel run //foo:bin` anything.

Removing this requirement is one of our top priorities. Follow issue
[#8240](https://github.com/bazelbuild/bazel/pull/8240) for updates.

<a name="sh-rules-without-bash"></a>
### `sh_binary` and `sh_*` rules, and `ctx.actions.run_shell()` without Bash

You need Bash to build and test `sh_*` rules, and to build and test Starlark
rules that use `ctx.actions.run_shell()` and `ctx.resolve_command()`. This
applies not only to rules in your project, but to rules in any of the external
repositories your project depends on (even transitively).

We may explore the option to use Windows Subsystem for Linux (WSL) to build
these rules, but as of 2019-05-07 it is not a priority for the Bazel-on-Windows
subteam.

## Setting environment variables

Environment variables you set in the Windows Command Prompt (`cmd.exe`) are only
set in that command prompt session. If you start a new `cmd.exe`, you need to
set the variables again. To always set the variables when `cmd.exe` starts, you
can add them to the User variables or System variables in the `Control Panel >
System Properties > Advanced > Environment Variables...` dialog box.

<a name="using"></a>
## Using Bazel on Windows

The first time you build any target, Bazel auto-configures the location of
Python and the Visual C++ compiler. If you need to auto-configure again, run
`bazel clean` then build a target.

You can also tell Bazel where to find the Python binary and the C++ compiler:

- use the [`--python_path=c:\path\to\python.exe`](command-line-reference.html#flag--python_path) flag for Python
- use the `BAZEL_VC` or the `BAZEL_VS` environment variable (they are *not* the same!).
  See the [Build C++ section](#build_cpp) below.

<a name="build_cpp"></a>
### Build C++

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

*   The `BAZEL_VS`, `BAZEL_VC` and `BAZEL_VC_FULL_VERSION` environment variable.

    Bazel tries to locate the C++ compiler the first time you build any
    target. To tell Bazel where the compiler is, you can set the
    following environment variables:

    For Visual Studio 2017 and 2019, set one of `BAZEL_VC` or `BAZEL_VS`. Additionally you may also set `BAZEL_VC_FULL_VERSION`.

    *   `BAZEL_VS` the Visual Studio installation directory

        ```
        set BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools
        ```

    *   `BAZEL_VC` the Visual C++ Build Tools installation directory
        ```
        set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC
        ```

    *   `BAZEL_VC_FULL_VERSION` (Optional) Only for Visual Studio 2017 and 2019, the full version
        number of your Visual C++ Build Tools. You can choose the exact Visual C++ Build Tools
        version via `BAZEL_VC_FULL_VERSION` if more than one version are installed, otherwise Bazel
        will choose the latest version.
        ```
        set BAZEL_VC_FULL_VERSION=14.16.27023
        ```

    For Visual Studio 2015 or older, set `BAZEL_VC` or `BAZEL_VS`. (`BAZEL_VC_FULL_VERSION` is not supported.)

    *   `BAZEL_VS` the Visual Studio installation directory

        ```
        set BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio 14.0
        ```

    *   `BAZEL_VC` the Visual C++ Build Tools installation directory
        ```
        set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC
        ```

*   The [Windows
    SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk).

    The Windows SDK contains header files and libraries you need when building
    Windows applications, including Bazel itself. By default, the latest Windows SDK installed will
    be used. You also can specify Windows SDK version by setting `BAZEL_WINSDK_FULL_VERSION`. You
    can use a full Windows 10 SDK number such as 10.0.10240.0, or specify 8.1 to use the Windows 8.1
    SDK (only one version of Windows 8.1 SDK is available). Please make sure you have the specified
    Windows SDK installed.

    **Requirement**: This is supported with VC 2017 and 2019. The standalone VC 2015 Build Tools doesn't
    support selecting Windows SDK, you'll need the full Visual Studio 2015 installation, otherwise
    `BAZEL_WINSDK_FULL_VERSION` will be ignored.

    ```
    set BAZEL_WINSDK_FULL_VERSION=10.0.10240.0
    ```

If everything is set up, you can build a C++ target now!

Try building a target from one of our [sample
projects](https://github.com/bazelbuild/bazel/tree/master/examples):

```
C:\projects\bazel> bazel build //examples/cpp:hello-world

C:\projects\bazel> bazel-bin\examples\cpp\hello-world.exe
```

To build and use Dynamically Linked Libraries (DLL files), see [this
example](https://github.com/bazelbuild/bazel/tree/master/examples/windows/dll).

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

---
layout: documentation
title: Installing Bazel on Windows
---

# <a name="windows"></a>Installing Bazel on Windows

Supported Windows platforms:

*   64 bit Windows 7 or higher, or equivalent Windows Server versions.

_Check
<a href="https://msdn.microsoft.com/en-us/library/windows/desktop/ms724832(v=vs.85).aspx">Microsoft's
Operating System Version table</a> to see if your OS is supported._

### 1. Install prerequisites (if not already installed)

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

*   [Microsoft Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145)

    This may already be installed on your system.

### 2. Install Bazel on Windows using one of the following methods:

*   [Download the binary (recommended)](#download-the-binary-recommended)
*   [Install using Chocolatey](#install-using-chocolatey)
*   [Compile Bazel from source](install-compile-source.html)

#### Download the binary (recommended)

Go to Bazel's [GitHub releases page](https://github.com/bazelbuild/bazel/releases)
and download the Windows binary<sup>1</sup>: `bazel-<version>-installer-windows-x86_64.sh`.

For convenience, move the binary to a directory that's on your `%PATH%`. This
way you can run Bazel by typing `bazel` in any directory, without typing out the
full path. That said, you may put the binary anywhere on your filesystem.

After you download the binary, you'll need additional
software and some setup in your environment to run Bazel. For details, see the
[Windows requirements](windows.html).

_<sup>1</sup>Note that Bazel includes an embedded JDK 8, which can be used even if a JDK is already installed. bazel-<version>-without-jdk-installer-windows-x86_64.sh is a version of the installer without embedded JDK 8. Only use this installer if you already have JDK 8 installed._

#### Install using Chocolatey

You can install the Bazel package using the [Chocolatey](https://chocolatey.org)
package manager:

```sh
choco install bazel
```

This command will install the latest available version of Bazel and most of
its dependencies, such as the MSYS2 shell. This will not install Visual C++
though.

See [Chocolatey installation and package maintenance
guide](https://bazel.build/windows-chocolatey-maintenance.html) for more
information about the Chocolatey package.

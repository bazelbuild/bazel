---
layout: documentation
title: Installing Bazel on Windows
---

# <a name="windows"></a>Installing Bazel on Windows

### Prerequisites

*   64 bit Windows 7 or newer, or 64 bit Windows Server 2008 R2 or newer

*   [MSYS2 shell](https://msys2.github.io/)

*   [Microsoft Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145)

    These are common DLLs that Bazel needs. You may already have them installed.

### Getting Bazel

[Download Bazel for Windows from GitHub](https://github.com/bazelbuild/bazel/releases).
Look for `bazel-<version>-windows-x86_64.exe`, e.g. `bazel-0.15.2-windows-x86_64.exe`.

**Tip:** Rename the binary to `bazel.exe` and move it to a directory on your
`%PATH%`, so you can run Bazel by typing `bazel` in any directory.

### Other ways to get Bazel

*   [Install Bazel using the Chocolatey package manager](#install-using-chocolatey)
*   [Compile Bazel from source](install-compile-source.html)

#### Install using Chocolatey

1.  Install the [Chocolatey](https://chocolatey.org) package manager

2.  Install the Bazel package:

        choco install bazel

    This command will install the latest available version of Bazel and
    its dependencies, such as the MSYS2 shell. This will not install Visual C++
    though.

See [Chocolatey installation and package maintenance
guide](https://bazel.build/windows-chocolatey-maintenance.html) for more
information about the Chocolatey package.

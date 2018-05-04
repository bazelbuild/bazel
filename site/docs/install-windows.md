---
layout: documentation
title: Installing Bazel on Windows
---

# <a name="windows"></a>Installing Bazel on Windows

### Prerequisites

*   64 bit Windows 7 or newer, or <a href="https://msdn.microsoft.com/en-us/library/windows/desktop/ms724832(v=vs.85).aspx">equivalent Windows Server versions</a>

*   [MSYS2 shell](https://msys2.github.io/)

*   [Microsoft Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145)

    These are common DLLs that Bazel needs. You may already have them installed.

### Getting Bazel

Download Bazel for Windows from our
[GitHub releases page](https://github.com/bazelbuild/bazel/releases).
Look for `bazel-<version>-windows-x86_64.exe`, e.g. `bazel-0.13.0-windows-x86_64.exe`.

**Tip:** For convenience, rename the downloaded binary to `bazel.exe` and move it to a directory
that's on your `%PATH%` or add its directory to your `%PATH%`. This way you can run Bazel by
typing `bazel` in any directory, without typing out the full path.

**Note:** Bazel includes an embedded JDK so you don't need to install one separately. If you want
to download Bazel without an embedded JDK, look for `bazel-<version>-without-jdk-windows-x86_64.exe`
on the release page. To use it, you must install JDK 8 separately (older or newer versions are not
supported), and set the `JAVA_HOME` environment variable, e.g.:
```
set JAVA_HOME=c:\Program Files\Java\jdk1.8.0_171
```
Pay attention not to use quotes (") around the path (like you would on Unixes).

### Other ways to get Bazel

You can also get Bazel by:
*   [Installing Bazel using Chocolatey](#install-using-chocolatey)
*   [Compiling Bazel from source](install-compile-source.html)

#### Install using Chocolatey

You can install the Bazel package using the [Chocolatey](https://chocolatey.org)
package manager:

```sh
choco install bazel
```

This command will install the latest available version of Bazel and
its dependencies, such as the MSYS2 shell. This will not install Visual C++
though.

See [Chocolatey installation and package maintenance
guide](https://bazel.build/windows-chocolatey-maintenance.html) for more
information about the Chocolatey package.

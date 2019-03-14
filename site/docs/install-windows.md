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

1.  [Download Bazel for Windows from GitHub](https://github.com/bazelbuild/bazel/releases).

    Look for `bazel-<version>-windows-x86_64.exe`, for example
    `bazel-0.16.1-windows-x86_64.exe`.

    **Tip:** Rename the binary to `bazel.exe` and move it to a directory on your
    `%PATH%` (for example to `c:\bazel`), so you can run Bazel by typing `bazel`
    in any directory.

1.  Edit environment variables.

    Open the "Environment Variables" dialog box from Control Panel or Start
    Menu, and add or edit the following variables under the "User variables"
    section:
    1.  **Edit `Path`**. Add new entries to the beginning of the list:
        *   The directory of `bazel.exe`. (Example: `c:\bazel`).
        *   The `usr\bin` directory of MSYS2. (Example: `c:\msys64\usr\bin`).
        *   If you will build **Python** code: the directory of `python.exe`.
            (Example: `c:\python3`).
    1.  **Add `BAZEL_SH`**. Its value must be the path to MSYS2 Bash.
        Example: `c:\msys64\usr\bin\bash.exe`
    1.  **Add `JAVA_HOME`** (if you will build **Java** code). Its value must be
        the directory where you installed the Java JDK 8, for example
        `C:\Program Files\Java\jdk1.8.0_152`. In order to use this with the
        default local_jdk javabase, it must be installed on a volume which
        Windows considers to be **local**, network mounted filesystems will not
        work.

    **None of these paths should contain spaces or non-ASCII characters.**


### Other ways to get Bazel

*   [Install Bazel using the Chocolatey package manager](#install-using-chocolatey)
*   [Install Bazel using the Scoop package manager](#install-using-scoop)
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

#### Install using Scoop

1.  Install the [Scoop](https://scoop.sh/) package manager using the following PowerShell command:

        iex (new-object net.webclient).downloadstring('https://get.scoop.sh')

2.  Install the Bazel package:

        scoop install bazel

See [Scoop installation and package maintenance
guide](https://bazel.build/windows-scoop-maintenance.html) for more
information about the Scoop package.

### Using Bazel

Once you have installed Bazel, see [Using Bazel on Windows](windows.html).

Project: /_project.yaml
Book: /_book.yaml

# Installing Bazel on Windows

{% include "_buttons.html" %}

This page describes the requirements and steps to install Bazel on Windows.
It also includes troubleshooting and other ways to install Bazel, such as
using Chocolatey or Scoop.

## Installing Bazel {:#installing-bazel}

This section covers the prerequisites, environment setup, and detailed
steps during installation on Windows.

### Check your system  {:#check-system}

Recommended: 64 bit Windows 10, version 1703 (Creators Update) or newer

To check your Windows version:

* Click the Start button.
* Type `winver` in the search box and press Enter.
* You should see the About Windows box with your Windows version information.

### Install the prerequisites {:#install-prerequisites}

*   [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170){: .external}

### Download Bazel {:#download-bazel}

*Recommended*: [Use Bazelisk](/install/bazelisk)


Alternatively you can:

*   [Download the Bazel binary (`bazel-{{ '<var>' }}version{{ '</var>' }}-windows-x86_64.exe`) from
 GitHub](https://github.com/bazelbuild/bazel/releases){: .external}.
*   [Install Bazel from Chocolatey](#chocolately)
*   [Install Bazel from Scoop](#scoop)
*   [Build Bazel from source](/install/compile-source)

### Set up your environment {:#set-environment}

To make Bazel easily accessible from command prompts or PowerShell by default, you can rename the Bazel binary to `bazel.exe` and add it to your default paths.

```posix-terminal
set PATH=%PATH%;{{ '<var>' }}path to the Bazel binary{{ '</var>' }}
```

You can also change your system `PATH` environment variable to make it permanent. Check out how to [set environment variables](/configure/windows#set-environment-variables).

### Done {:#done}

"Success: You've installed Bazel."

To check the installation is correct, try to run:

```posix-terminal
bazel {{ '<var>' }}version{{ '</var>' }}
```

Next, you can check out more tips and guidance here:

*   [Installing compilers and language runtimes](#install-compilers)
*   [Troubleshooting](#troubleshooting)
*   [Best practices on Windows](/configure/windows#best-practices)
*   [Tutorials](/start/#tutorials)

## Installing compilers and language runtimes {:#install-compilers}

Depending on which languages you want to build, you will need:

*   [MSYS2 x86_64](https://www.msys2.org/){: .external}

    MSYS2 is a software distro and building platform for Windows. It contains Bash and common Unix
    tools (like `grep`, `tar`, `git`).

    You will need MSYS2 to build, test, or run targets that depend on Bash. Typically these are
    `genrule`, `sh_binary`, `sh_test`, but there may be more (such as Starlark rules). Bazel shows an
    error if a build target needs Bash but Bazel could not locate it.

*   Common MSYS2 packages

    You will likely need these to build and run targets that depend on Bash. MSYS2 does not install
    these tools by default, so you need to install them manually. Projects that depend on Bash tools in `PATH` need this step (for example TensorFlow).

    Open the MSYS2 terminal and run this command:

    ```posix-terminal
    pacman -S zip unzip patch diffutils git
    ```

    Optional: If you want to use Bazel from CMD or Powershell and still be able
    to use Bash tools, make sure to add
    `{{ '<var>' }}MSYS2_INSTALL_PATH{{ '</var>' }}/usr/bin` to your
    `PATH` environment variable.

*   [Build Tools for Visual Studio 2019](https://aka.ms/buildtools){:#install-vc}

    You will need this to build C++ code on Windows.

    Also supported:

    *   Visual C++ Build Tools 2017 (or newer) and Windows 10 SDK

*   [Java SE Development Kit 11 (JDK) for Windows x64](https://www.oracle.com/java/technologies/javase-jdk11-downloads.html){: .external}{:#install-jdk}

    You will need this to build Java code on Windows.

    Also supported: Java 8, 9, and 10

*   [Python 3.6 for Windows x86-64](https://www.python.org/downloads/windows/){:#install-python}

    You will need this to build Python code on Windows.

    Also supported: Python 2.7 or newer for Windows x86-64

## Troubleshooting {:#troubleshooting}

### Bazel does not find Bash or bash.exe

**Possible reasons**:

*   you installed MSYS2 not under the default install path

*   you installed MSYS2 i686 instead of MSYS2 x86\_64

*   you installed MSYS instead of MSYS2

**Solution**:

Ensure you installed MSYS2 x86\_64.

If that doesn't help:

1.  Go to Start Menu &gt; Settings.

2.  Find the setting "Edit environment variables for your account"

3.  Look at the list on the top ("User variables for &lt;username&gt;"), and click the "New..."
    button below it.

4.  For "Variable name", enter `BAZEL_SH`

5.  Click "Browse File..."

6.  Navigate to the MSYS2 directory, then to `usr\bin` below it.

    For example, this might be `C:\msys64\usr\bin` on your system.

7.  Select the `bash.exe` or `bash` file and click OK

8.  The "Variable value" field now has the path to `bash.exe`. Click OK to close the window.

9.  Done.

    If you open a new cmd.exe or PowerShell terminal and run Bazel now, it will find Bash.

### Bazel does not find Visual Studio or Visual C++

**Possible reasons**:

*   you installed multiple versions of Visual Studio

*   you installed and removed various versions of Visual Studio

*   you installed various versions of the Windows SDK

*   you installed Visual Studio not under the default install path

**Solution**:

1.  Go to Start Menu &gt; Settings.

2.  Find the setting "Edit environment variables for your account"

3.  Look at the list on the top ("User variables for &lt;username&gt;"), and click the "New..."
    button below it.

4.  For "Variable name", enter `BAZEL_VC`

5.  Click "Browse Directory..."

6.  Navigate to the `VC` directory of Visual Studio.

    For example, this might be `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC`
    on your system.

7.  Select the `VC` folder and click OK

8.  The "Variable value" field now has the path to `VC`. Click OK to close the window.

9.  Done.

    If you open a new cmd.exe or PowerShell terminal and run Bazel now, it will find Visual C++.

## Other ways to install Bazel {:#install-options}

### Using Chocolatey {:#chocolately}

1.  Install the [Chocolatey](https://chocolatey.org) package manager

2.  Install the Bazel package:

    ```posix-terminal
    choco install bazel
    ```

    This command will install the latest available version of Bazel and
    its dependencies, such as the MSYS2 shell. This will not install Visual C++
    though.

See [Chocolatey installation and package maintenance
guide](/contribute/windows-chocolatey-maintenance) for more
information about the Chocolatey package.

### Using Scoop {:#scoop}

1.  Install the [Scoop](https://scoop.sh/) package manager using the following PowerShell command:

    ```posix-terminal
    iex (new-object net.webclient).downloadstring('https://get.scoop.sh')
    ```

2.  Install the Bazel package:

    ```posix-terminal
    scoop install bazel
    ```

See [Scoop installation and package maintenance
guide](/contribute/windows-scoop-maintenance) for more
information about the Scoop package.

### Build from source {:#build-from-source}

To build Bazel from scratch instead of installing, see [Compiling from source](/install/compile-source).

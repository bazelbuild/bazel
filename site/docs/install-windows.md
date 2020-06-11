---
layout: documentation
title: Installing Bazel on Windows
---

# Installing Bazel on Windows

## Installing Bazel

### Step 1: Check your system

Recommended: 64 bit Windows 10, version 1703 (Creators Update) or newer

To check your Windows version:
* Click the Start button.
* Type `winver` in the search box and press Enter.
* You should see the About Windows box with your Windows version information.

Also supported:

*   64 bit Windows 7 or newer

*   64 bit Windows Server 2008 R2 or newer

### Step 2: Install the prerequisites

*   [Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145)

### Step 3: Download Bazel

[Download the Bazel binary (<code>bazel-&lt;version&gt;-windows-x86_64.exe</code>) from
 GitHub](https://github.com/bazelbuild/bazel/releases).

Alternatively you can:

*   [Download Bazelisk](https://github.com/bazelbuild/bazelisk) instead of Bazel. Bazelisk is a
    Bazel launcher that ensures you always use the latest Bazel release.
*   [Install Bazel from Chocolatey](#using-chocolatey)
*   [Install Bazel from Scoop](#using-scoop)
*   [Build Bazel from source](install-compile-source.html)

### Step 4: Set up your environment

To make Bazel easily accessible from command prompts or PowerShell by default, you can rename the Bazel binary to `bazel.exe` and add it to your default paths.

```batch
set PATH=%PATH%;<path to the Bazel binary>
```

You can also change your system `PATH` environment variable to make it permanent. Check out how to [set environment variables](windows.html#setting-environment-variables).

### Step 5: Done

**You have successfully installed Bazel.**
To check the installation is correct, try to run:
```batch
bazel version
```

Next, you can check out more tips and guidance here:

*   [Installing compilers and language runtimes](#installing-compilers-and-language-runtimes)
*   [Troubleshooting](#troubleshooting)
*   [Best practices on Windows](windows.html#best-practices)
*   [Tutorials](getting-started.html#tutorials)

---

## Installing compilers and language runtimes

Depending on which languages you want to build, you will need:

*   [MSYS2 x86_64](https://www.msys2.org/)

    MSYS2 is a software distro and building platform for Windows. It contains Bash and common Unix
    tools (like `grep`, `tar`, `git`).

    You will need MSYS2 to build, test, or run targets that depend on Bash. Typically these are
    `genrule`, `sh_binary`, `sh_test`, but there may be more (e.g. Starlark rules). Bazel shows an
    error if a build target needs Bash but Bazel could not locate it.

*   Common MSYS2 packages

    You will likely need these to build and run targets that depend on Bash.  MSYS2 does not install
    these tools by default, so you need to install them manually. Projects that depend on Bash tools in `PATH` need this step (for example TensorFlow).

    Open the MSYS2 terminal and run this command:

    ```bash
    pacman -S zip unzip patch diffutils git
    ```
    Optional: If you want to use Bazel from CMD or Powershell and still be able to use Bash tools, make sure to add `<MSYS2_INSTALL_PATH>/usr/bin` to your `PATH` environment variable.

<a name="install-vc"></a>
*   [Build Tools for Visual Studio 2019](https://aka.ms/buildtools)

    You will need this to build C++ code on Windows.

    Also supported:

    *   Visual Studio 2015 (or newer) with Visual C++ and Windows 10 SDK

    *   Visual C++ Build Tools 2015 (or newer) and Windows 10 SDK

<a name="install-jdk"></a>
*   [Java SE Development Kit 11 (JDK) for Windows x64](https://www.oracle.com/java/technologies/javase-jdk11-downloads.html)

    You will need this to build Java code on Windows.

    Also supported: Java 8, 9, and 10

<a name="install-python"></a>
*   [Python 3.6 for Windows x86-64](https://www.python.org/downloads/windows/)

    You will need this to build Python code on Windows.

    Also supported: Python 2.7 or newer for Windows x86-64

---

## Troubleshooting

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

## Other ways to install Bazel

### Using Chocolatey

1.  Install the [Chocolatey](https://chocolatey.org) package manager

2.  Install the Bazel package:

        choco install bazel

    This command will install the latest available version of Bazel and
    its dependencies, such as the MSYS2 shell. This will not install Visual C++
    though.

See [Chocolatey installation and package maintenance
guide](https://bazel.build/windows-chocolatey-maintenance.html) for more
information about the Chocolatey package.

### Using Scoop

1.  Install the [Scoop](https://scoop.sh/) package manager using the following PowerShell command:

        iex (new-object net.webclient).downloadstring('https://get.scoop.sh')

2.  Install the Bazel package:

        scoop install bazel

See [Scoop installation and package maintenance
guide](https://bazel.build/windows-scoop-maintenance.html) for more
information about the Scoop package.

### Build from source

See [here](install-compile-source.html).

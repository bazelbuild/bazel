---
layout: documentation
title: Installing Bazel on Windows
---

# Installing Bazel on Windows

## 1. Check your system

Recommended: 64 bit Windows 10, version 1703 (Creators Update) or newer, enable "Developer Mode".

<!-- Developer mode: for symlink support. -->

Also supported:

*   64 bit Windows 7 or newer

*   64 bit Windows Server 2008 R2 or newer

## 2. Install the prerequisites

*   [Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145)

*   [MSYS2 x86_64](https://www.msys2.org/)

    You should use the default install path.

*   MSYS2 packages

    Open the MSYS2 terminal, and run this command:

    ```
    pacman -S zip unzip patch diffutils git
    ```

## 3. Download Bazel

[Download the Bazel binary (<code>bazel-&lt;version&gt;-windows-x86_64.exe</code>) from
 GitHub](https://github.com/bazelbuild/bazel/releases).

Recommended: rename this binary to `bazel.exe` and move it to a directory on the `PATH`.

Alternatively you can:

*   [Install from Chocolatey](#install-using-chocolatey) (see below)
*   [Install from Scoop](#install-using-scoop) (see below)
*   [Build from source](install-compile-source.html)

## 4. Optional: install compilers

**You can skip this step. Bazel can work without these programs, but you may need them.**

We recommend installing:

*   [Build Tools for Visual Studio 2019](https://aka.ms/buildtools)

    Make sure you install the C++ build tools with the Windows 10 SDK.

    You will need this to build C++ code on Windows.

    Also supported:

    *   Visual Studio 2015 (or newer) with Visual C++ and Windows 10 SDK

    *   Visual C++ Build Tools 2015 (or newer) and Windows 10 SDK

*   Java SE Development Kit 10 (JDK) for Windows x64

    You will need this to build Java code on Windows.

    Also supported: Java 8, 9

*   [Python 2.7 for Windows x86-64](https://www.python.org/downloads/windows/)

    You will need this to build Python code on Windows.

    Also supported: Python 3 or newer for Windows x86-64

## 5. Done

**You have successfully installed Bazel.**

Troubleshooting: see the "Appendix" &gt; "Troubleshooting" section below.

Tutorials: see the "Tutorials" section on the left navigation panel.

---

# Appendix

Table of contents:

*   [Troubleshooting](#troubleshooting)
    *   [Problem: Bazel does not find Bash or bash.exe](#problem-bazel-does-not-find-bash-or-bashexe)
    *   [Problem: Bazel does not find Visual Studio or Visual C++](#problem-bazel-does-not-find-visual-studio-or-visual-c)

*   [Other ways to install Bazel](#other-ways-to-install-bazel)
    *   [Install from Chocolatey](#install-using-chocolatey)
    *   [Install from Scoop](#install-using-scoop)
    *   [Build from source](install-compile-source.html)

## Troubleshooting

### Problem: Bazel does not find Bash or bash.exe

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

### Problem: Bazel does not find Visual Studio or Visual C++

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

### Install using Chocolatey

1.  Install the [Chocolatey](https://chocolatey.org) package manager

2.  Install the Bazel package:

        choco install bazel

    This command will install the latest available version of Bazel and
    its dependencies, such as the MSYS2 shell. This will not install Visual C++
    though.

See [Chocolatey installation and package maintenance
guide](https://bazel.build/windows-chocolatey-maintenance.html) for more
information about the Chocolatey package.

### Install using Scoop

1.  Install the [Scoop](https://scoop.sh/) package manager using the following PowerShell command:

        iex (new-object net.webclient).downloadstring('https://get.scoop.sh')

2.  Install the Bazel package:

        scoop install bazel

See [Scoop installation and package maintenance
guide](https://bazel.build/windows-scoop-maintenance.html) for more
information about the Scoop package.

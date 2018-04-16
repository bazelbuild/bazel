---
layout: documentation
title: Installing Bazel on Windows
---

# <a name="windows"></a>Installing Bazel on Windows

Supported Windows platforms:

*   64-bit Windows 7 or higher, or equivalent Windows Server versions.

### Installation

1.  Download and install [MSYS2](https://msys2.github.io/).

2.  Download the [latest Bazel binary](https://github.com/bazelbuild/bazel/releases).

    No installation required.

    Optional: rename the binary to `bazel.exe` and move it to a directory that's on your `PATH`.

3.  Try Bazel.

    *    Start PowerShell or the Command Prompt (`cmd.exe`).

    *    Go to the directory where you downloaded Bazel:
         ```sh
         cd c:\Users\myusername\Downloads
         ```

    *    Run `bazel version`:
         ```sh
         bazel version
         ```
         Example output:
         ```sh
         Extracting Bazel installation...
         .
         Build label: 0.12.0
         Build target: bazel-out/x64_windows-opt/bin/src/main/java/com/google/devtools/build/lib/bazel/BazelServer_deploy.jar
         Build time: Tue Aug 4 02:44:15 +50246 (1523462006655)
         Build timestamp: 1523462006655
         Build timestamp as int: 1523462006655
         ```

#### Other ways to get Bazel

*   [Compile Bazel from source](install-compile-source.html)

*   Install using [Chocolatey](https://chocolatey.org)

    Install the latest Bazel and dependencies:

    ```sh
    choco install bazel
    ```

    See [Chocolatey package maintenance guide](https://bazel.build/windows-chocolatey-maintenance.html) for more
information.

#### Troubleshooting

*   Bazel won't start.

    -   **Incompatible Windows version**.
    
        Check the <a href="https://msdn.microsoft.com/en-us/library/windows/desktop/ms724832(v=vs.85).aspx">version table</a>. Bazel requires 64-bit Windows 7 or higher, or equivalent Windows Server versions. 32-bit Windows is not supported.

    -   **"The application was unable to start correctly (0xc000007b)." error**
    
        Install the [Microsoft Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145).
        
    -   **MSVCP140.DLL is missing** or **VCRUNTIME140.DLL is missing**
    
        Install the [Microsoft Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145).


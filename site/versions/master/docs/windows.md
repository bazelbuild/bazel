---
layout: documentation
title: Windows
---

Windows support is highly experimental. Known issues are [marked with
label "Windows"](https://github.com/bazelbuild/bazel/issues?q=is%3Aissue+is%3Aopen+label%3A%22category%3A+multi-platform+%3E+windows%22)
on github issues.


Installing Bazel on Windows
===========================

You can install using the [chocolatey](https://chocolatey.org) package manager:

```shell
choco install bazel
```

This will install the latest available version of bazel, and dependencies.


Using Bazel on Windows
======================

Bazel currently supports building C++ targets and Java targets on Windows.

### Build C++

To build C++ targets, you will need:

* [Visual Studio](https://www.visualstudio.com/)
<br/>We are using MSVC as the native C++ toolchain, so please ensure you have Visual
Studio installed with the Visual C++ components
(which is NOT the default installation type of Visual Studio).

* [Python 2.7](https://www.python.org/downloads/)
<br/>Currently, we use python wrapper scripts to call the actual MSVC compiler, so
please make sure Python is installed and its location is added into PATH.
It's also a good idea to set BAZEL\_PYTHON environment variable to tell Bazel
where python is.
<br/>For example: `export BAZEL_PYTHON=C:/Python27/python.exe`

Bazel will auto-configure the location of Visual Studio and Python at the first
time you build any target.
If you need to auto-configure again, just run `bazel clean` then build a target.

If everything is set up, you can build C++ target now! However, since MSVC
toolchain is not default on Windows yet, you should use flag
`--cpu=x64_windows_msvc` to enable it like this:

```bash
$ bazel build --cpu=x64_windows_msvc examples/cpp:hello-world
$ ./bazel-bin/examples/cpp/hello-world.exe
$ bazel run --cpu=x64_windows_msvc examples/cpp:hello-world
```

### Build Java

Building Java targets works well on Windows, no special configuration is needed.
Just try:

```bash
$ bazel build examples/java-native/src/main/java/com/example/myproject:hello-world
$ ./bazel-bin/examples/java-native/src/main/java/com/example/myproject/hello-world
$ bazel run examples/java-native/src/main/java/com/example/myproject:hello-world
```


Building Bazel on Windows
=========================

We currently support only 64 bit Windows 7 or higher and we compile Bazel as a msys2 binary.

To bootstrap Bazel on Windows, you will need:

*    Java JDK 8 or later
*    [Visual Studio](https://www.visualstudio.com/) (Community Edition is okay)
*    [msys2](https://msys2.github.io/) (need to be installed at
     ``C:\tools\msys64\``).
*    Several msys2 packages. Use the ``pacman`` command to install them:
     ``pacman -Syuu gcc git curl zip unzip zlib-devel``

To build Bazel:

*    Open the msys2 shell.
*    Clone the Bazel git repository as normal.
*    Set the environment variables:

```bash
export JAVA_HOME="$(ls -d C:/Program\ Files/Java/jdk* | sort | tail -n 1)"
export BAZEL_SH=c:/tools/msys64/usr/bin/bash.exe
```

*     Run ``compile.sh`` in Bazel directory.
*     If all works fine, bazel will be built at ``output\bazel.exe``.


Maintaining Bazel Chocolatey package on Windows
===============================================

### Prerequisites

You need:
* [chocolatey package manager](https://chocolatey.org) installed
* (to publish) a chocolatey API key granting you permission to publish the `bazel` package
* (to publish) to have set up that API key for the chocolatey source locally via `choco apikey -k <your key here> -s https://chocolatey.org/`

### Build

Compile bazel with msys2 shell and `compile.sh`.

```powershell
pushd scripts/packages/chocolatey
  ./build.ps1 -version 0.3.1 -isRelease
popd
```

Should result in `scripts/packages/chocolatey/bazel.<version>.nupkg` being created.

#### Test

0. Build the package (without `isRelease`)
  * run a webserver (`python -m SimpleHTTPServer` in `scripts/packages/chocolatey` is convenient and starts one on `http://localhost:8000`)
  * adjust `chocolateyinstall.ps1` so that the `$url` and `$url64` parameters point to `http://localhost:8000/bazel_0.3.1_windows_x86_64.zip`
0. Test the install

    The `test.ps1` should install the package cleanly (and error if it did not install cleanly), then tell you what to do next.
    
    In a new (msys2) shell
    ```shell
    bazel version
    ```
    should result in that version, with executable from PATH.

0. Test the uninstall

    ```shell
    choco uninstall bazel
    # should remove bazel from the system - c:/tools/bazel should be deleted
    ```

Chocolatey's moderation process automates checks here.

### Publish

```shell
choco push bazel.x.y.z.nupkg --source https://chocolatey.org/
```

Chocolatey.org will then run automated checks and respond to the 

---
layout: documentation
title: Windows Chocolatey maintenance
---

Installing Bazel on Windows
===========================

You can install the unofficial package using the [chocolatey](https://chocolatey.org) package manager:

```sh
choco install bazel
```

This will install the latest available version of bazel, and dependencies.

This package is experimental; please provide feedback (`@petemounce` in issue tracker).


Maintaining Bazel Chocolatey package on Windows
===============================================

### Prerequisites

You need:
* [chocolatey package manager](https://chocolatey.org) installed
* (to publish) a chocolatey API key granting you permission to publish the `bazel` package
  * [@petemounce](https://github.com/petemounce) currently maintains this unofficial package.
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
    ```sh
    bazel version
    ```
    should result in that version, with executable from PATH.

0. Test the uninstall

    ```sh
    choco uninstall bazel
    # should remove bazel from the system - c:/tools/bazel should be deleted
    ```

Chocolatey's moderation process automates checks here.

### Publish

```sh
choco push bazel.x.y.z.nupkg --source https://chocolatey.org/
```

Chocolatey.org will then run automated checks and respond to the push via email to the maintainers.

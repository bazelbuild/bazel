---
layout: documentation
title: Installing Bazel on openSUSE
---

# Installing Bazel on openSUSE Tumbleweed & Leap

NOTE: The Bazel team does not maintain openSUSE support. For issues
using bazel on openSUSE please file ticket at bugzilla.opensuse.org.

The commands below must be run either via `sudo` or while logged in as `root`.

## Installing Bazel on openSUSE Tumbleweed

1. Run the following commands to install the package, if you need a specific 
    version, you should use specific `bazelXXX` package, otherwise, just `bazel` 
    is enough:

    ```bash
    zypper install bazelXXX
    ```

## Installing Bazel on openSUSE Leap

1. Run the following commands to install the package:

    ```bash
    zypper install bazel
    ```

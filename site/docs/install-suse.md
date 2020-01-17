---
layout: documentation
title: Installing Bazel on openSUSE
---

# Installing Bazel on openSUSE Tumbleweed & Leap

The commands below must be run either via `sudo` or while logged in as `root`.

## Installing Bazel on openSUSE Tumbleweed

1. Run the following commands to install the package, if you need a specific 
    version, you should use specific `bazelXXX` package, otherwise, just `bazel` 
    is enough:

    ```bash
    zypper install bazelXXX
    ```

## Installing Bazel on openSUSE Leap

1. Run the following commands to add the Bazel repository and install the
    package, if you need a specific version, you should use specific `bazelXXX` 
    package, otherwise, just `bazel` is enough:

    ```bash
    zypper ar -r https://download.opensuse.org/repositories/devel:/tools:/building/openSUSE_Leap_15.1/devel:tools:building.repo
    zypper refresh
    zypper install bazelXXX
    ```

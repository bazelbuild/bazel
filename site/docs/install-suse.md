---
layout: documentation
title: Installing Bazel on OpenSUSE
---

# Installing Bazel on OpenSUSE

[OpenSUSE Package](https://software.opensuse.org/package/bazel).

The commands below must be run either via `sudo` or while logged in as `root`.

## Installing Bazel on OpenSUSE

1. Run the following commands to add the Bazel repository and install the
    package:

    ```bash
    zypper addrepo https://download.opensuse.org/repositories/openSUSE:Factory/standard/openSUSE:Factory.repo
    zypper refresh
    zypper install bazel
    ```


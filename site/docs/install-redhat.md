---
layout: documentation
title: Installing Bazel on Fedora and CentOS
---

# Installing Bazel on Fedora and CentOS

The Bazel team does not provide official packages for Fedora and CentOS.
Vincent Batts ([@vbatts](https://github.com/vbatts)) generously maintains
unofficial packages on
[Fedora COPR](https://copr.fedorainfracloud.org/coprs/vbatts/bazel/).

Note that all commands will need to be run with appropriate permissions, most
typically by using `sudo` or by logging in as root.

## Installing Bazel on Fedora 25+

1. The [DNF](https://fedoraproject.org/wiki/DNF) package manager can install
    Bazel from the [COPR](https://copr.fedorainfracloud.org/) repository. Install
    the `copr` plugin for DNF if you have not already done so.

    ```bash
    dnf install dnf-plugins-core
    ```

2. Run the following commands to add the Bazel repository and install the
    package:

    ```bash
    dnf copr enable vbatts/bazel
    dnf install bazel
    ```

## Installing Bazel on CentOS 7

1. Download the corresponding `.repo` file from
    [Fedora COPR](https://copr.fedorainfracloud.org/coprs/vbatts/bazel/repo/epel-7/vbatts-bazel-epel-7.repo)
    and copy it to `/etc/yum.repos.d/`.

2. Run the following command:

    ```bash
    yum install bazel
    ```

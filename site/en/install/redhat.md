Project: /_project.yaml
Book: /_book.yaml

# Installing Bazel on Fedora and CentOS

{% include "_buttons.html" %}

This page describes how to install Bazel on Fedora and CentOS.

The Bazel team does not provide official packages for Fedora and CentOS.
Vincent Batts ([@vbatts](https://github.com/vbatts){: .external}) generously maintains
unofficial packages on
[Fedora COPR](https://copr.fedorainfracloud.org/coprs/vbatts/bazel/){: .external}.

The commands below must be run either via `sudo` or while logged in as `root`.

Add `--allowerasing` when installing an upgrade from a previous major
version of the Bazel package.

[The Bazelisk installer](/install/bazelisk) is an alternative to package installation.

## Installing on Fedora 25+ {:#installing-fedora}

1. The [DNF](https://fedoraproject.org/wiki/DNF){: .external} package manager can
   install Bazel from the [COPR](https://copr.fedorainfracloud.org/){: .external} repository.
   Install the `copr` plugin for DNF if you have not already done so.

    ```posix-terminal
    dnf install dnf-plugins-core
    ```

2. Run the following commands to add the Bazel repository and install the
   package:

    ```posix-terminal
    dnf copr enable vbatts/bazel

    dnf install bazel4
    ```

## Installing on CentOS 7 {:#installing-centos}

1. Download the corresponding `.repo` file from
   [Fedora COPR](https://copr.fedorainfracloud.org/coprs/vbatts/bazel/repo/epel-7/vbatts-bazel-epel-7.repo){: .external}
   and copy it to `/etc/yum.repos.d/`.

2. Run the following command:

    ```posix-terminal
    yum install bazel4
    ```

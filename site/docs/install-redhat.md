---
layout: documentation
title: Installing Bazel on Fedora and CentOS
---

# Installing Bazel on Fedora and CentOS

The Bazel team does not provide official packages for Fedora and CentOS.
Vincent Batts [@vbatts](https://github.com/vbatts) generously maintains
unofficial packages on
[Fedora COPR](https://copr.fedorainfracloud.org/coprs/vbatts/bazel/).

## Installing Bazel on Fedora 25, 26

1. Install `dnf` and the `copr` plugin if you have not already done so.

2. Run the following commands:

   ```bash
   dnf copr enable vbatts/bazel
   dnf install bazel
   ```

## Installing Bazel on CentOS 7

1. Download the corresponding `.repo` file from [Fedora COPR](https://copr.fedorainfracloud.org/coprs/vbatts/bazel/)
   and copy it to `/etc/yum.repos.d/`.

2. Run the following command:

   ```bash
   yum install bazel
   ```

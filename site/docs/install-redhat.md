---
layout: documentation
title: Installing Bazel on Fedora and CentOS
---
# Install Bazel on Fedora and CentOS
The Bazel team does not provide official packages for Fedora and CentOS.
Vincent Batts [@vbatts](https://github.com/vbatts) generously maintains
unofficial packages on [Fedora Copr].

## Install Bazel on Fedora 25, 26

You need to have `dnf` and the `copr` plugin installed.

```bash
dnf copr enable vbatts/bazel
dnf install bazel
```

## Install Bazel on CentOS 7

Download the corresponding `.repo` file from [Fedora Copr]
and copy it into `/etc/yum.repos.d/`.

Then type

```bash
yum install bazel
```

[Fedora Copr]:
https://copr.fedorainfracloud.org/coprs/vbatts/bazel/

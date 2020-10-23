---
layout: documentation
title: Installing Bazel on openSUSE
---

# Installing Bazel on openSUSE Tumbleweed & Leap

NOTE: The Bazel team does not officially maintain openSUSE support. For issues
using Bazel on openSUSE please file a ticket at [bugzilla.opensuse.org](https://bugzilla.opensuse.org/).

Packages are provided for openSUSE Tumbleweed and Leap. You can find all
available Bazel versions via openSUSE's [software search](https://software.opensuse.org/search?utf8=%E2%9C%93&baseproject=ALL&q=bazel).

The commands below must be run either via `sudo` or while logged in as `root`.

## Installing Bazel on openSUSE

Run the following commands to install the package. If you need a specific
version, you can install it via the specific `bazelXXX` package, otherwise,
just `bazel` is enough:

```bash
# Install the latest Bazel version.
zypper install bazel

# Alternatively: Install Bazel 1.2.
zypper install bazel1.2
```

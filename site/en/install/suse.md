Project: /_project.yaml
Book: /_book.yaml

# Installing Bazel on openSUSE Tumbleweed & Leap

{% include "_buttons.html" %}

This page describes how to install Bazel on openSUSE Tumbleweed and Leap.

`NOTE:` The Bazel team does not officially maintain openSUSE support. For issues
using Bazel on openSUSE please file a ticket at [bugzilla.opensuse.org](https://bugzilla.opensuse.org/){: .external}.

Packages are provided for openSUSE Tumbleweed and Leap. You can find all
available Bazel versions via openSUSE's [software search](https://software.opensuse.org/search?utf8=%E2%9C%93&baseproject=ALL&q=bazel){: .external}.

The commands below must be run either via `sudo` or while logged in as `root`.

## Installing Bazel on openSUSE {:#install-opensuse}

Run the following commands to install the package. If you need a specific
version, you can install it via the specific `bazelXXX` package, otherwise,
just `bazel` is enough:

To install the latest version of Bazel, run:

```posix-terminal
zypper install bazel
```

You can also install a specific version of Bazel by specifying the package
version with `bazel{{ '<var>' }}version{{ '</var>' }}`. For example, to install
Bazel 4.2, run:

```posix-terminal
zypper install bazel4.2
```

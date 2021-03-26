---
layout: documentation
title: Installing Bazel on macOS
category: getting-started
---

<h1 id="mac-os-x">Installing Bazel on macOS</h1>

This page describes how to install Bazel on macOS and set up your environment.

You can install Bazel on macOS using one of the following methods:

*   [Use the binary installer (recommended)](#install-with-installer-mac-os-x)
*   [Use Homebrew](#install-on-mac-os-x-homebrew)
*   [Use Bazelisk](install-bazelisk.md)
*   [Compile Bazel from source](install-compile-source.md)

Bazel comes with two completion scripts. After installing Bazel, you can:

*   Access the [bash completion script](completion.md#bash)
*   Install the [zsh completion script](completion.md#zsh)

<h2 id="install-with-installer-mac-os-x">Installing using the binary installer</h2>

The binary installers are on Bazel's
[GitHub releases page](https://github.com/bazelbuild/bazel/releases).

The installer contains the Bazel binary.<sup>1</sup> Some additional libraries
must also be installed for Bazel to work.

### Step 1: Install Xcode command line tools

If you don't intend to use `ios_*` rules, it is sufficient to install the Xcode
command line tools package by using `xcode-select`:

```shell
xcode-select --install
```

Otherwise, for `ios_*` rule support, you must have Xcode 6.1 or later with iOS
SDK 8.1 installed on your system.

Download Xcode from the
[App Store](https://apps.apple.com/us/app/xcode/id497799835) or the
[Apple Developer site](https://developer.apple.com/download/more/?=xcode).

Once Xcode is installed, accept the license agreement for all users with the
following command:

```shell
sudo xcodebuild -license accept
```

### Step 2: Download the Bazel installer

Next, download the Bazel binary installer named
`bazel-<version>-installer-darwin-x86_64.sh` from the
[Bazel releases page on GitHub](https://github.com/bazelbuild/bazel/releases).

Note: **on macOS Catalina**, due to Apple's new app notarization requirements,
you will need to download the installer from the terminal using `curl`:

```shell
# Example installing version `3.2.0`. Replace the version below as appropriate.
export BAZEL_VERSION=3.2.0
curl -fLO "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh"
```

This is a temporary workaround until the macOS release flow supports
notarization ([#9304](https://github.com/bazelbuild/bazel/issues/9304)).

### Step 3: Run the installer

Run the Bazel installer as follows:

```shell
chmod +x "bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh"
./bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh --user
```

The `--user` flag installs Bazel to the `$HOME/bin` directory on your system and
sets the `.bazelrc` path to `$HOME/.bazelrc`. Use the `--help` command to see
additional installation options.

If you are **on macOS Catalina** and get an error that _**“bazel-real” cannot be
opened because the developer cannot be verified**_, you will need to re-download
the installer from the terminal using `curl` as a workaround; see Step 2 above.

### Step 4: Set up your environment

If you ran the Bazel installer with the `--user` flag as above, the Bazel
executable is installed in your `$HOME/bin` directory. It's a good idea to add
this directory to your default paths, as follows:

```shell
export PATH="$PATH:$HOME/bin"
```

You can also add this command to your `~/.bashrc`, `~/.zshrc`, or `~/.profile`
file.

All set! You can confirm Bazel is installed successfully by running the
following command:

```shell
bazel --version
```
To update to a newer release of Bazel, download and install the desired version.

<h2 id="install-on-mac-os-x-homebrew">Installing using Homebrew</h2>

### Step 1: Install Homebrew on macOS

Install Homebrew (a one-time step):

```shell
/bin/bash -c "$(curl -fsSL \
https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
```

### Step 2: Install Bazel via Homebrew

Install the Bazel package via Homebrew as follows:

```shell
$ brew install bazel
```

All set! You can confirm Bazel is installed successfully by running the
following command:

```shell
$ bazel --version
```

Once installed, you can upgrade to a newer version of Bazel using the
following command:

```shell
$ brew upgrade bazel
```

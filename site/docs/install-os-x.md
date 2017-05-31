---
layout: documentation
title: Installing Bazel on macOS
---

# <a name="mac-os-x"></a>Install Bazel on macOS (OS X)

> Note: Bazel release 0.5.0 contains a bug in the compiler detection on macOS which
> requires Xcode and the iOS tooling to be installed
> ([corresponding issue #3063](https://github.com/bazelbuild/bazel/issues/3063)).
> If you had Command Line Tools installed, you also need to switch to Xcode using
> `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`.

Install Bazel on macOS (OS X) using one of the following methods:

*   [Use Homebrew (recommended)](#install-on-mac-os-x-homebrew)
*   [Use the binary installer](#install-with-installer-mac-os-x)
*   [Compile Bazel from source](install-compile-source.md)

Bazel comes with two completion scripts. After installing Bazel, you can:

*   access the [bash completion script](install.md)
*   install the [zsh completion script](install.md)

## <a name="install-on-mac-os-x-homebrew"></a>Install using Homebrew

### 1. Install JDK 8

JDK 8 can be downloaded from [Oracle's JDK
Page](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html).

Look for "Mac OS X" under "Java SE Development Kit". This will download a DMG
image with an install wizard.

### 2. Install Homebrew on macOS (OS X)

Installing Homebrew is a one-time setup:

```bash
/usr/bin/ruby -e "$(curl -fsSL
https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

### 3. Install Bazel Homebrew Package

```bash
brew install bazel
```

You are all set. You can confirm Bazel is installed successfully by running
`bazel version`.

You can later upgrade to newer version of Bazel with `brew upgrade bazel`.

## <a name="install-with-installer-mac-os-x"></a>Install using binary installer

The binary installers are on Bazel's [GitHub releases page](https://github.com/bazelbuild/bazel/releases).

The installer contains the Bazel binary and the required JDK. Some additional
libraries must also be installed for Bazel to work.

### 1. Install XCode command line tools

Xcode can be downloaded from the [Apple Developer
Site](https://developer.apple.com/xcode/downloads/) (this link redirects to
their App Store).

For `objc_*` and `ios_*` rule support, you must have Xcode 6.1 or later with iOS
SDK 8.1 installed on your system.

Once XCode is installed you can trigger signing the license with the following
command:

```
sudo gcc --version
```

### 2. Download the Bazel installer

Go to Bazel's [GitHub releases page](https://github.com/bazelbuild/bazel/releases).

Download the binary installer `bazel-0.5.0-installer-darwin-x86_64.sh`. This
installer contains the Bazel binary and the required JDK, and can be used even
if a JDK is already installed.

Note that `bazel-0.5.0-without-jdk-installer-darwin-x86_64.sh` is a version of
the installer without embedded JDK 8. Only use this installer if you already
have JDK 8 installed.

Note that two other versions of the installer exist:
*   `bazel-0.5.0-without-jdk-installer-darwin-x86_64.sh`: version without
    embedded JDK 8. Only use this installer if you already have JDK 8 installed.
*   `bazel-0.5.0-jdk7-installer-darwin-x86_64.sh`: last release compatible
    with JDK 7.

### 3. Run the installer

Run the installer:

<pre>
chmod +x bazel-0.5.0-installer-darwin-x86_64.sh
./bazel-0.5.0-installer-darwin-x86_64.sh --user
</pre>

The `--user` flag installs Bazel to the `$HOME/bin` directory on your system and
sets the `.bazelrc` path to `$HOME/.bazelrc`. Use the `--help` command to see
additional installation options.

### 4. Set up your environment

If you ran the Bazel installer with the `--user` flag as above, the Bazel
executable is installed in your `$HOME/bin` directory. It's a good idea to add
this directory to your default paths, as follows:

```bash
export PATH="$PATH:$HOME/bin"
```

You can also add this command to your `~/.bashrc` file.

You are all set. You can confirm Bazel is installed successfully by running
```bash
bazel version
```

Once installed, you can upgrade to a newer version of Bazel with:

```bash
sudo apt-get upgrade bazel
```

---
layout: documentation
title: Installing Bazel on macOS
---

# <a name="mac-os-x"></a>Installing Bazel on macOS

Install Bazel on macOS using one of the following methods:

*   [Use the binary installer (recommended)](#install-with-installer-mac-os-x)
*   [Use Homebrew](#install-on-mac-os-x-homebrew)
*   [Compile Bazel from source](install-compile-source.md)

Bazel comes with two completion scripts. After installing Bazel, you can:

*   Access the [bash completion script](install.md)
*   Install the [zsh completion script](install.md)

## <a name="install-with-installer-mac-os-x"></a>Installing using binary installer

The binary installers are on Bazel's [GitHub releases page](https://github.com/bazelbuild/bazel/releases).

The installer contains the Bazel binary.<sup>1</sup> Some additional libraries must also be
installed for Bazel to work.

### Step 1: Install Xcode command line tools

Xcode can be downloaded from the [Apple Developer
Site](https://developer.apple.com/xcode/downloads/) (this link redirects to
their App Store).

For `objc_*` and `ios_*` rule support, you must have Xcode 6.1 or later with iOS
SDK 8.1 installed on your system.

Once Xcode is installed, accept the license agreement for all users with the following command:

```
sudo xcodebuild -license accept
```

### Step 2: Download the Bazel installer

Next, download the Bazel binary installer named `bazel-<version>-installer-darwin-x86_64.sh` from the [Bazel releases page on GitHub](https://github.com/bazelbuild/bazel/releases).

### Step 3: Run the installer

Run the Bazel installer as follows:

<pre>
chmod +x bazel-<version>-installer-darwin-x86_64.sh
./bazel-<version>-installer-darwin-x86_64.sh --user
</pre>

The `--user` flag installs Bazel to the `$HOME/bin` directory on your system and
sets the `.bazelrc` path to `$HOME/.bazelrc`. Use the `--help` command to see
additional installation options.

### Step 4: Set up your environment

If you ran the Bazel installer with the `--user` flag as above, the Bazel
executable is installed in your `$HOME/bin` directory. It's a good idea to add
this directory to your default paths, as follows:

```bash
export PATH="$PATH:$HOME/bin"
```

You can also add this command to your `~/.bashrc` or `~/.profile` file.

All set! You can confirm Bazel is installed successfully by running the following command:

```bash
bazel version
```
To update to a newer release of Bazel, download and install the desired version.

**Note:** Bazel includes an embedded JDK, which can be used even if a JDK is already
installed. `bazel-<version>-without-jdk-installer-linux-x86_64.sh` is a version of the installer
without an embedded JDK. Only use this installer if you already have JDK 8 installed. Later JDK
versions are not supported.

## <a name="install-on-mac-os-x-homebrew"></a>Installing using Homebrew

### Step 1: Install the JDK

Download the JDK from [Oracle's JDK Page](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html). Look for "macOS" under "Java SE Development Kit" and download JDK version 8.

### Step 2: Install Homebrew on macOS

Install Homebrew (a one-time step):

```bash
/usr/bin/ruby -e "$(curl -fsSL \
https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

### Step 3: Install the Bazel Homebrew package

Install the Bazel package via Homebrew as follows:

```bash
brew install bazel
```

All set! You can confirm Bazel is installed successfully by running the following command:

```bash
bazel version
```

Once installed, you can upgrade to a newer version of Bazel using the following command:

```bash
brew upgrade bazel
```

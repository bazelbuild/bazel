---
layout: documentation
title: Installing Bazel on macOS
---

<h1 id="mac-os-x">Installing Bazel on macOS</h1>

Install Bazel on macOS using one of the following methods:

*   [Use the binary installer (recommended)](#install-with-installer-mac-os-x)
*   [Use Homebrew](#install-on-mac-os-x-homebrew)
*   [Compile Bazel from source](install-compile-source.md)

Bazel comes with two completion scripts. After installing Bazel, you can:

*   Access the [bash completion script](completion.md#bash)
*   Install the [zsh completion script](completion.md#zsh)

<h2 id="install-with-installer-mac-os-x">Installing using binary installer</h2>

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

```
chmod +x bazel-<version>-installer-darwin-x86_64.sh
./bazel-<version>-installer-darwin-x86_64.sh --user
```

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

<h2 id="install-on-mac-os-x-homebrew">Installing using Homebrew</h2>

### Step 1: Install Homebrew on macOS

Install Homebrew (a one-time step):

```bash
/usr/bin/ruby -e "$(curl -fsSL \
https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

### Step 2: Install the Bazel Homebrew package

_Please note that if your system has the Bazel package from homebrew core installed you first
need to uninstall it by typing `brew uninstall bazel`_

Install the Bazel package via Homebrew as follows:

```bash
brew tap bazelbuild/tap
brew tap-pin bazelbuild/tap
brew install bazelbuild/tap/bazel
```

All set! You can confirm Bazel is installed successfully by running the following command:

```bash
bazel version
```

Once installed, you can upgrade to a newer version of Bazel using the following command:

```bash
brew upgrade bazelbuild/tap/bazel
```



---
layout: documentation
title: Installing Bazel on macOS
---

# <a name="mac-os-x"></a>Install Bazel on macOS (OS X)

**Known issue on MacOS**

Bazel release 0.5.0 contains a bug in the compiler detection on macOS which
requires Xcode and the iOS tooling to be installed
([corresponding issue #3063](https://github.com/bazelbuild/bazel/issues/3063)).
If you had Command Line Tools installed, you also need to switch to Xcode using
`sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`.

Install Bazel on macOS (OS X) using one of the following methods:

*   [Use Homebrew](#install-on-mac-os-x-homebrew)
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

## <a name="install-with-installer-mac-os-x"></a>Install with installer

We provide binary installers on our
<a href="https://github.com/bazelbuild/bazel/releases">GitHub releases page</a>

The installer only contains Bazel binary, some additional libraries are required
to be installed on the machine to work.

### 1. Install JDK 8

JDK 8 can be downloaded from [Oracle's JDK
Page](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html).
Look for "Mac OS X" under "Java SE Development Kit". This will download a DMG
image with an install wizard.

### 2. Install XCode command line tools

Xcode can be downloaded from the [Apple Developer
Site](https://developer.apple.com/xcode/downloads/), which will redirect to the
App Store.

For `objc_*` and `ios_*` rule support, you must have Xcode 6.1 or later with iOS
SDK 8.1 installed on your system.

Once XCode is installed you can trigger signing the license with the following
command:

```
sudo gcc --version
```

### 3. Download Bazel

Download the [Bazel installer](https://github.com/bazelbuild/bazel/releases) for
your operating system.

### 4. Run the installer

Run the installer:

<pre>
chmod +x bazel-&lt;version&gt;-installer-&lt;os&gt;.sh
./bazel-&lt;version&gt;-installer-&lt;os&gt;.sh --user
</pre>

The `--user` flag installs Bazel to the `$HOME/bin` directory on your system and
sets the `.bazelrc` path to `$HOME/.bazelrc`. Use the `--help` command to see
additional installation options.

### 5. Set up your environment

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

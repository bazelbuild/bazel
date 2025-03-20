Project: /_project.yaml
Book: /_book.yaml

# Installing Bazel on macOS

{% include "_buttons.html" %}

This page describes how to install Bazel on macOS and set up your environment.

You can install Bazel on macOS using one of the following methods:

*   *Recommended*: [Use Bazelisk](/install/bazelisk)
*   [Use Homebrew](#install-on-mac-os-x-homebrew)
*   [Use the binary installer](#install-with-installer-mac-os-x)
*   [Compile Bazel from source](/install/compile-source)

Bazel comes with two completion scripts. After installing Bazel, you can:

*   Access the [bash completion script](/install/completion#bash)
*   Install the [zsh completion script](/install/completion#zsh)

<h2 id="install-on-mac-os-x-homebrew">Installing using Homebrew</h2>

### Step 1: Install Homebrew on macOS

Install [Homebrew](https://brew.sh/) (a one-time step):

```posix-terminal
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install Bazel via Homebrew

Install the Bazel package via Homebrew as follows:

```posix-terminal
brew install bazel
```

All set! You can confirm Bazel is installed successfully by running the
following command:

```posix-terminal
bazel --version
```

Once installed, you can upgrade to a newer version of Bazel using the
following command:

```posix-terminal
brew upgrade bazel
```

<h2 id="install-with-installer-mac-os-x">Installing using the binary installer</h2>

The binary installers are on Bazel's
[GitHub releases page](https://github.com/bazelbuild/bazel/releases){: .external}.

The installer contains the Bazel binary. Some additional libraries
must also be installed for Bazel to work.

### Step 1: Install Xcode command line tools

If you don't intend to use `ios_*` rules, it is sufficient to install the Xcode
command line tools package by using `xcode-select`:

```posix-terminal
xcode-select --install
```

Otherwise, for `ios_*` rule support, you must have Xcode 6.1 or later with iOS
SDK 8.1 installed on your system.

Download Xcode from the
[App Store](https://apps.apple.com/us/app/xcode/id497799835){: .external} or the
[Apple Developer site](https://developer.apple.com/download/more/?=xcode){: .external}.

Once Xcode is installed, accept the license agreement for all users with the
following command:

```posix-terminal
sudo xcodebuild -license accept
```

### Step 2: Download the Bazel installer

Next, download the Bazel binary installer named
`bazel-<version>-installer-darwin-x86_64.sh` from the
[Bazel releases page on GitHub](https://github.com/bazelbuild/bazel/releases){: .external}.

**On macOS Catalina or newer (macOS >= 11)**, due to Apple's new app signing requirements,
you need to download the installer from the terminal using `curl`, replacing
the version variable with the Bazel version you want to download:

```posix-terminal
export BAZEL_VERSION=5.2.0

curl -fLO "https://github.com/bazelbuild/bazel/releases/download/{{ '<var>' }}$BAZEL_VERSION{{ '</var>' }}/bazel-{{ '<var>' }}$BAZEL_VERSION{{ '</var>' }}-installer-darwin-x86_64.sh"
```

This is a temporary workaround until the macOS release flow supports
signing ([#9304](https://github.com/bazelbuild/bazel/issues/9304){: .external}).

### Step 3: Run the installer

Run the Bazel installer as follows:

```posix-terminal
chmod +x "bazel-{{ '<var>' }}$BAZEL_VERSION{{ '</var>' }}-installer-darwin-x86_64.sh"

./bazel-{{ '<var>' }}$BAZEL_VERSION{{ '</var>' }}-installer-darwin-x86_64.sh --user
```

The `--user` flag installs Bazel to the `$HOME/bin` directory on your system and
sets the `.bazelrc` path to `$HOME/.bazelrc`. Use the `--help` command to see
additional installation options.

If you are **on macOS Catalina or newer (macOS >= 11)** and get an error that _**“bazel-real” cannot be
opened because the developer cannot be verified**_, you need to re-download
the installer from the terminal using `curl` as a workaround; see Step 2 above.

### Step 4: Set up your environment

If you ran the Bazel installer with the `--user` flag as above, the Bazel
executable is installed in your `{{ '<var>' }}HOME{{ '</var>' }}/bin` directory.
It's a good idea to add this directory to your default paths, as follows:

```posix-terminal
export PATH="{{ '<var>' }}PATH{{ '</var>' }}:{{ '<var>' }}HOME{{ '</var>' }}/bin"
```

You can also add this command to your `~/.bashrc`, `~/.zshrc`, or `~/.profile`
file.

All set! You can confirm Bazel is installed successfully by running the
following command:

```posix-terminal
bazel --version
```
To update to a newer release of Bazel, download and install the desired version.


---
layout: documentation
title: Installing Bazel on Ubuntu
---

<h1 id="ubuntu">Installing Bazel on Ubuntu</h1>

Supported Ubuntu Linux platforms:

*   18.04 (LTS)
*   16.04 (LTS)

Bazel will probably work fine on other Ubuntu releases and Debian stretch and
above, but we currently do not test this on Bazel's CI and thus can't promise
it.

Install Bazel on Ubuntu using one of the following methods:

*   [Use our custom APT repository (recommended)](#install-on-ubuntu)
*   [Use the binary installer](#install-with-installer-ubuntu)
*   [Compile Bazel from source](install-compile-source.md)

Bazel comes with two completion scripts. After installing Bazel, you can:

*   Access the [bash completion script](completion.md#bash)
*   Install the [zsh completion script](completion.md#zsh)

<h2 id="install-on-ubuntu"> Using Bazel's apt repository</h2>

### Step 1: Add Bazel distribution URI as a package source

**Note:** This is a one-time setup step.

```bash
sudo apt install curl gnupg
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
```

The component name "jdk1.8" is kept for legacy reasons only and doesn't relate
to supported or included JDK versions anymore. In the past, when Bazel did not
yet bundle a private JRE, we had two release versions, one compatible with JDK 7
and one with JDK 8. However, since we dropped Java 7 support and started
bundling a private runtime, Bazel releases are Java version agnostic. Changing
the "jdk1.8" component name would break existing users of the repo though.

### Step 2: Install and update Bazel

```bash
sudo apt update && sudo apt install bazel
```

Once installed, you can upgrade to a newer version of Bazel as part of your normal system updates:

```bash
sudo apt update && sudo apt full-upgrade
```

The `bazel` package will always install the latest stable version of Bazel. You
can install specific, older versions of Bazel in addition to the latest one like
this:

```bash
sudo apt install bazel-1.0.0
```

This will install Bazel 1.0.0 as `/usr/bin/bazel-1.0.0` on your system. This
can be useful if you need a specific version of Bazel to build a project, e.g.
because it uses a `.bazelversion` file to explicitly state with which Bazel
version it should be built.

### Step 3: Install a JDK (optional)

Bazel includes a private, bundled JRE as its runtime and doesn't require you to
install any specific version of Java.

However, if you want to build Java code using Bazel, you have to install a JDK.

```bash
# Ubuntu 16.04 (LTS) uses OpenJDK 8 by default:
sudo apt install openjdk-8-jdk

# Ubuntu 18.04 (LTS) uses OpenJDK 11 by default:
sudo apt install openjdk-11-jdk
```

<h2 id="install-with-installer-ubuntu">Using the binary installer</h2>

While we generally recommend to use the apt repository, the binary installer can
be useful in case you don't have admin permissions on your machine or can't add
custom repositories.

The binary installers can be downloaded from Bazel's [GitHub releases page](https://github.com/bazelbuild/bazel/releases).

The installer contains the Bazel binary and extracts it into your `$HOME/bin`
folder. Some additional libraries must be installed manually for Bazel to work.

### Step 1: Install required packages

Bazel needs a C++ compiler and unzip / zip in order to work:

```bash
sudo apt install g++ unzip zip
```

If you want to build Java code using Bazel, install a JDK:

```bash
# Ubuntu 16.04 (LTS) uses OpenJDK 8 by default:
sudo apt-get install openjdk-8-jdk

# Ubuntu 18.04 (LTS) uses OpenJDK 11 by default:
sudo apt-get install openjdk-11-jdk
```

### Step 2: Run the installer

Next, download the Bazel binary installer named `bazel-<version>-installer-linux-x86_64.sh`
from the [Bazel releases page on GitHub](https://github.com/bazelbuild/bazel/releases).

Run it as follows:

```bash
chmod +x bazel-<version>-installer-linux-x86_64.sh
./bazel-<version>-installer-linux-x86_64.sh --user
```

The `--user` flag installs Bazel to the `$HOME/bin` directory on your system and
sets the `.bazelrc` path to `$HOME/.bazelrc`. Use the `--help` command to see
additional installation options.

### Step 3: Set up your environment

If you ran the Bazel installer with the `--user` flag as above, the Bazel
executable is installed in your `$HOME/bin` directory. It's a good idea to add
this directory to your default paths, as follows:

```bash
export PATH="$PATH:$HOME/bin"
```

You can also add this command to your `~/.bashrc` or `~/.zshrc` file to make it
permanent.

---
layout: documentation
title: Installing Bazel on Ubuntu
---

# <a name="ubuntu"></a>Installing Bazel on Ubuntu

Supported Ubuntu Linux platforms:

*   16.04 (LTS)
*   14.04 (LTS)

Install Bazel on Ubuntu using one of the following methods:

*   [Use the binary installer (recommended)](#install-with-installer-ubuntu)
*   [Use our custom APT repository](#install-on-ubuntu)
*   [Compile Bazel from source](install-compile-source.md)

Bazel comes with two completion scripts. After installing Bazel, you can:

*   Access the [bash completion script](completion.md#bash)
*   Install the [zsh completion script](completion.md#zsh)

## <a name="install-with-installer-ubuntu"></a>Installing using binary installer

The binary installers are on Bazel's [GitHub releases page](https://github.com/bazelbuild/bazel/releases).

The installer contains the Bazel binary<sup>1</sup>. Some additional libraries must
also be installed for Bazel to work.

### Step 1: Install required packages

First, install the prerequisites: `pkg-config`, `zip`, `g++`, `zlib1g-dev`, `unzip`, and `python`.

```bash
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python
```

### Step 2: Download Bazel

Next, download the Bazel binary installer named `bazel-<version>-installer-linux-x86_64.sh`
from the [Bazel releases page on GitHub](https://github.com/bazelbuild/bazel/releases).

### Step 3: Run the installer

Run the Bazel installer as follows:

```bash
chmod +x bazel-<version>-installer-linux-x86_64.sh
./bazel-<version>-installer-linux-x86_64.sh --user
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

You can also add this command to your `~/.bashrc` file.

## <a name="install-on-ubuntu"></a> Using Bazel custom APT repository

### Step 1: Install the JDK

Install JDK 8:

```bash
sudo apt-get install openjdk-8-jdk
```

On Ubuntu 14.04 LTS you must use a PPA:

```bash
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update && sudo apt-get install oracle-java8-installer
```

### Step 2: Add Bazel distribution URI as a package source

**Note:** This is a one-time setup step.

```bash
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
```

If you want to install the testing version of Bazel, replace `stable` with `testing`.

### Step 3: Install and update Bazel

```bash
sudo apt-get update && sudo apt-get install bazel
```

Once installed, you can upgrade to a newer version of Bazel with the following command:

```bash
sudo apt-get upgrade bazel
```

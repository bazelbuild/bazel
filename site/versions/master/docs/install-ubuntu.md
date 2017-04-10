---
layout: documentation
title: Installing Bazel on Ubuntu
---

# <a name="ubuntu"></a>Install Bazel on Ubuntu

Supported Ubuntu Linux platforms:

*   16.04 (LTS)
*   15.10
*   14.04 (LTS)

Install Bazel on Ubuntu using one of the following methods:

*   [Use our custom APT repository (recommended)](#install-on-ubuntu)
*   [Use the binary installer](#install-with-installer-ubuntu)
*   [Compile Bazel from source](#install-compile-source.md)

Bazel comes with two completion scripts. After installing Bazel, you can:

*   access the [bash completion script](install.md)
*   install the [zsh completion script](install.md)

## <a name="install-on-ubuntu"></a> Using Bazel custom APT repository (recommended)

### 1. Add Bazel distribution URI as a package source (one time setup)

```
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
```

If you want to use the JDK 7, please replace `jdk1.8` with `jdk1.7` and if you
want to install the testing version of Bazel, replace `stable` with `testing`.

### 2. Install and update Bazel

```bash
sudo apt-get update && sudo apt-get install bazel
```

Once installed, you can upgrade to newer version of Bazel with:

```bash
sudo apt-get upgrade bazel
```

## <a name="install-with-installer-ubuntu"></a>Install with installer

We provide binary installers on our
<a href="https://github.com/bazelbuild/bazel/releases">GitHub releases page</a>

The installer only contains Bazel binary, some additional libraries are required
to be installed on the machine to work.

### 1. Install JDK 8

To install OpenJDK 8:

```
sudo apt-get install openjdk-8-jdk
```

### 2. Install other required packages

```
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip
```

### 3. Download Bazel

Download the [Bazel installer](https://github.com/bazelbuild/bazel/releases) for
your operating system.

### 4. Run the installer

Run the installer:

```bash
chmod +x bazel-<version>-installer-<os>.sh
./bazel-<version>-installer-<os>.sh --user
```

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

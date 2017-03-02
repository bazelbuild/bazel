---
layout: documentation
title: Installing Bazel
---

# Installing Bazel

Supported platforms:

*   [Ubuntu Linux (Wily 15.10 and Trusty 14.04 LTS)](#ubuntu)
*   [Mac OS X](#mac-os-x)
*   [Windows (highly experimental)](#windows)

For other platforms, you can try to [compile from source](#compiling-from-source).

Required Java version:

*   Java JDK 8 or later ([JDK 7](#jdk7) is still supported
    but deprecated).

Extras:

*   [Bash completion](#bash)
*   [zsh completion](#zsh)

For more information on using Bazel, see [Getting
started](getting-started.html).


## <a name="ubuntu"></a>Ubuntu

Install Bazel on Ubuntu using one of the following methods:

  * [Using our custom APT repository](#install-on-ubuntu)
  * [Using binary installer](#install-with-installer-ubuntu)
  * [Compiling Bazel from source](#compiling-from-source)

### <a name="install-on-ubuntu"></a> Using Bazel custom APT repository (recommended)

#### 1. Install JDK 8

If you are running **Ubuntu Wily (15.10)**, you can skip this step.
But for **Ubuntu Trusty (14.04 LTS)** users, since OpenJDK 8 is not available on Trusty, please install Oracle JDK 8:

```
$ sudo add-apt-repository ppa:webupd8team/java
$ sudo apt-get update
$ sudo apt-get install oracle-java8-installer
```

Note: You might need to `sudo apt-get install software-properties-common` if you don't have the `add-apt-repository` command. See [here](http://manpages.ubuntu.com/manpages/wily/man1/add-apt-repository.1.html).

#### 2. Add Bazel distribution URI as a package source (one time setup)

```
$ echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
$ curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
```

If you want to use the JDK 7, please replace `jdk1.8` with `jdk1.7` and if you want to install the testing version of Bazel, replace `stable` with `testing`.

#### 3. Install and update Bazel

`$ sudo apt-get update && sudo apt-get install bazel`

Once installed, you can upgrade to newer version of Bazel with:

`$ sudo apt-get upgrade bazel`

### <a name="install-with-installer-ubuntu"></a>Install with Installer

We provide binary installers on our
<a href="https://github.com/bazelbuild/bazel/releases">GitHub releases page</a>

The installer only contains Bazel binary, some additional libraries are required to be installed on the machine to work.


#### 1. Install JDK 8

**Ubuntu Trusty (14.04 LTS).** OpenJDK 8 is not available on Trusty. To
install Oracle JDK 8:

```
$ sudo add-apt-repository ppa:webupd8team/java
$ sudo apt-get update
$ sudo apt-get install oracle-java8-installer
```

Note: You might need to `sudo apt-get install software-properties-common` if you don't have the `add-apt-repository` command. See [here](http://manpages.ubuntu.com/manpages/wily/man1/add-apt-repository.1.html).

**Ubuntu Wily (15.10).** To install OpenJDK 8:

```
$ sudo apt-get install openjdk-8-jdk
```

#### 2. Install other required packages

```
$ sudo apt-get install pkg-config zip g++ zlib1g-dev unzip
```

#### 3. Download Bazel

Download the [Bazel installer](https://github.com/bazelbuild/bazel/releases) for
your operating system.

#### 4. Run the installer

Run the installer:

<pre>
$ chmod +x bazel-<em>version</em>-installer-<em>os</em>.sh
$ ./bazel-<em>version</em>-installer-<em>os</em>.sh --user
</pre>

The `--user` flag installs Bazel to the `$HOME/bin` directory on your
system and sets the `.bazelrc` path to `$HOME/.bazelrc`. Use the `--help`
command to see additional installation options.

#### 5. Set up your environment

If you ran the Bazel installer with the `--user` flag as above, the Bazel
executable is installed in your `$HOME/bin` directory. It's a good idea to add
this directory to your default paths, as follows:

```bash
$ export PATH="$PATH:$HOME/bin"
```

You can also add this command to your `~/.bashrc` file.



## <a name="mac-os-x"></a>Mac OS X

Install Bazel on Mac OS X using one of the following methods:

  * [Install using Homebrew](#install-on-mac-os-x-homebrew)
  * [Install with installer](#install-with-installer-mac-os-x)
  * [Compiling Bazel from source](#compiling-from-source)


### <a name="install-on-mac-os-x-homebrew"></a>Install using Homebrew

#### 1. Install JDK 8

JDK 8 can be downloaded from
[Oracle's JDK Page](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html).
Look for "Mac OS X x64" under "Java SE Development Kit". This will download a
DMG image with an install wizard.

#### 2. Install Homebrew on Mac OS X (one time setup)

`$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`

#### 3. Install Bazel Homebrew Package

`$ brew install bazel`

You are all set. You can confirm Bazel is installed successfully by running `bazel version`.

You can later upgrade to newer version of Bazel with `brew upgrade bazel`.

### <a name="install-with-installer-mac-os-x"></a>Install with installer

We provide binary installers on our
<a href="https://github.com/bazelbuild/bazel/releases">GitHub releases page</a>

The installer only contains Bazel binary, some additional libraries are required to be installed on the machine to work.

#### 1. Install JDK 8

JDK 8 can be downloaded from
[Oracle's JDK Page](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html).
Look for "Mac OS X x64" under "Java SE Development Kit". This will download a
DMG image with an install wizard.

#### 2. Install XCode command line tools

Xcode can be downloaded from the
[Apple Developer Site](https://developer.apple.com/xcode/downloads/), which will
redirect to the App Store.

For `objc_*` and `ios_*` rule support, you must have Xcode 6.1 or later with
iOS SDK 8.1 installed on your system.

Once XCode is installed you can trigger signing the license with the following
command:

```
$ sudo gcc --version
```

#### 3. Download Bazel

Download the [Bazel installer](https://github.com/bazelbuild/bazel/releases) for
your operating system.

#### 4. Run the installer

Run the installer:

<pre>
$ chmod +x bazel-<em>version</em>-installer-<em>os</em>.sh
$ ./bazel-<em>version</em>-installer-<em>os</em>.sh --user
</pre>

The `--user` flag installs Bazel to the `$HOME/bin` directory on your
system and sets the `.bazelrc` path to `$HOME/.bazelrc`. Use the `--help`
command to see additional installation options.

#### 5. Set up your environment

If you ran the Bazel installer with the `--user` flag as above, the Bazel
executable is installed in your `$HOME/bin` directory. It's a good idea to add
this directory to your default paths, as follows:

```bash
$ export PATH="$PATH:$HOME/bin"
```

You can also add this command to your `~/.bashrc` file.

You are all set. You can confirm Bazel is installed successfully by running `bazel version`.

## <a name="windows"></a>Windows

Windows support is highly experimental. Known issues are [marked with
label "Windows"](https://github.com/bazelbuild/bazel/issues?q=is%3Aissue+is%3Aopen+label%3A%22category%3A+multi-platform+%3E+windows%22)
on GitHub issues.

We currently support only 64 bit Windows 7 or higher and we compile Bazel as a
MSYS2 binary.

Install Bazel on Windows using one of the following methods:

  * [Using Chocolatey](#install-on-windows-chocolatey)
  * [Using binary distribution](#download-binary-windows)
  * [Compiling Bazel from source](#compiling-from-source) -- make sure
    your machine meets the [requirements](windows.md#requirements)


### <a name="install-on-windows-chocolatey"></a>Install on Windows using Chocolatey

You can install the unofficial package using the
[chocolatey](https://chocolatey.org) package manager:

```sh
choco install bazel
```

This will install the latest available version of bazel, and dependencies.

This package is experimental; please provide feedback to `@petemounce` in GitHub
issue tracker. See the [Chocolatey installation and package
maintenance](windows-chocolatey-maintenance.md) guide for more information.


### <a name="download-binary-windows"></a>Download a precompiled binary

We provide binary versions on our
<a href="https://github.com/bazelbuild/bazel/releases">GitHub releases page</a>

This is merely the Bazel binary. You'll need additional software (e.g. msys2
shell of the right version) and some setup in your environment to run Bazel.
See these requirements on our [Windows page](windows.md#requirements).


## <a name="compiling-from-source"></a>Compiling from source

The standard way of compiling a release version of Bazel from source
is to use a distribution archive. Download `bazel-<VERSION>-dist.zip`
from
the [release page](https://github.com/bazelbuild/bazel/releases) for
the desired version. We recommend to also verify the signature made by our
[release key](https://bazel.build/bazel-release.pub.gpg) 48457EE0.

Unzip the archive and call `bash ./compile.sh`; this will create a
bazel binary in `output/bazel`. This binary is self-contained,
so it can be copied to a directory on the PATH (e.g.,
`/usr/local/bin`) or used in-place.

###<a name="compiling-from-source-issues"></a>Known issues when compiling from source

**On Windows:**

* version 0.4.4 and below: `compile.sh` may fail right after start with an error
  like this:

    ```
    File not found - *.jar
    no error prone jar
    ```

    Workaround is to run this (and add it to your `~/.bashrc`):

    ```
    export PATH="/bin:/usr/bin:$PATH"
    ```

* version 0.4.3 and below: `compile.sh` may fail fairly early with many Java
  compilation errors. The errors look similar to:

    ```
    C:\...\bazel_VR1HFY7x\src\com\google\devtools\build\lib\remote\ExecuteServiceGrpc.java:11: error: package io.grpc.stub does not exist
    import static io.grpc.stub.ServerCalls.asyncUnaryCall;
                              ^
    ```

    This is caused by a bug in one of the bootstrap scripts
    (`scripts/bootstrap/compile.sh`). Manually apply this one-line fix if you want
    to build Bazel purely from source (without using an existing Bazel binary):
    [5402993a5e9065984a42eca2132ec56ca3aa456f]( https://github.com/bazelbuild/bazel/commit/5402993a5e9065984a42eca2132ec56ca3aa456f).

* version 0.3.2 and below:
  [github issue #1919](https://github.com/bazelbuild/bazel/issues/1919)


## <a name="jdk7"></a>Using Bazel with JDK 7 (deprecated)

Bazel version _0.1.0_ runs without any change with JDK 7. However, future
version will stop supporting JDK 7 when our CI cannot build for it anymore.
The installer for JDK 7 for Bazel versions after _0.1.0_ is labeled
<pre>
./bazel-<em>version</em>-jdk7-installer-<em>os</em>.sh
</pre>
If you wish to use JDK 7, follow the same steps as for JDK 8 but with the _jdk7_ installer or using a different APT repository as described [here](#1-add-bazel-distribution-uri-as-a-package-source-one-time-setup).

## <a name="bash"></a>Getting bash completion

Bazel comes with a bash completion script, which the installer copies into the
`bin` directory. If you ran the installer with `--user`, this will be
`$HOME/.bazel/bin`. If you ran the installer as root, this will be
`/usr/local/bazel/bin`.

Copy the `bazel-complete.bash` script to your completion folder
(`/etc/bash_completion.d` directory under Ubuntu). If you don't have a
completion folder, you can copy it wherever suits you and insert
`source /path/to/bazel-complete.bash` in your `~/.bashrc` file (under OS X, put
it in your `~/.bash_profile` file).

If you built Bazel from source, the bash completion target is in the `//scripts`
package:

1. Build it with Bazel: `bazel build //scripts:bazel-complete.bash`.
2. Copy the script `bazel-bin/scripts/bazel-complete.bash` to one of the
   locations described above.

## <a name="zsh"></a>Getting zsh completion

Bazel also comes with a zsh completion script. To install it:

1. Add this script to a directory on your $fpath:

    ```
    fpath[1,0]=~/.zsh/completion/
    mkdir -p ~/.zsh/completion/
    cp scripts/zsh_completion/_bazel ~/.zsh/completion
    ```

    You may have to call `rm -f ~/.zcompdump; compinit`
    the first time to make it work.

2. Optionally, add the following to your .zshrc.

    ```
    # This way the completion script does not have to parse Bazel's options
    # repeatedly.  The directory in cache-path must be created manually.
    zstyle ':completion:*' use-cache on
    zstyle ':completion:*' cache-path ~/.zsh/cache
    ```

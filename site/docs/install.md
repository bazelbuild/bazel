---
layout: documentation
---

# Installing Bazel

## System Requirements

Supported platforms:

*   Ubuntu Linux (Wily 15.10 and Trusty 14.04 LTS)
*   Mac OS X

Java:

*   Java JDK 8 or later ([JDK 7](#jdk7) is still supported
    but deprecated).

## Install dependencies

### Ubuntu

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
$ sudo apt-get install openjdk-8-jdk openjdk-8-source
```

#### 2. Install required packages

```
$ sudo apt-get install pkg-config zip g++ zlib1g-dev unzip
```

### Mac OS X

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

Once XCode is installed you can trigger the license signature with the following
command:

```
$ sudo gcc --version
```

## Download Bazel

Download the [Bazel installer](https://github.com/bazelbuild/bazel/releases) for
your operating system.

## Run the installer

Run the installer:

<pre>
$ chmod +x bazel-<em>version</em>-installer-<em>os</em>.sh
$ ./bazel-<em>version</em>-installer-<em>os</em>.sh --user
</pre>

The `--user` flag installs Bazel to the `$HOME/bin` directory on your
system and sets the `.bazelrc` path to `$HOME/.bazelrc`. Use the `--help`
command to see additional installation options.

## Set up your environment

If you ran the Bazel installer with the `--user` flag as above, the Bazel
executable is installed in your `$HOME/bin` directory. It's a good idea to add
this directory to your default paths, as follows:

```bash
$ export PATH="$PATH:$HOME/bin"
```

You can also add this command to your `~/.bashrc` file.

## <a name="jdk7"></a>Using Bazel with JDK 7 (deprecated)

Bazel version _0.1.0_ runs without any change with JDK 7. However, future
version will stop supporting JDK 7 when our CI cannot build for it anymore.
The installer for JDK 7 for Bazel versions after _0.1.0_ is labeled
`./install-<em>version</em>-<em>os</em>-<b>jdk7</b>.sh`. If you wish to use JDK
7, follow the same steps as for JDK 8 but with the _jdk7_ installer.

### Getting bash completion

Bazel comes with a bash completion script. To install it:

1. Build it with Bazel: `bazel build //scripts:bazel-complete.bash`.
2. Copy the script `bazel-bin/scripts/bazel-complete.bash` to your
   completion folder (`/etc/bash_completion.d` directory under Ubuntu).
   If you don't have a completion folder, you can copy it wherever suits
   you and simply insert `source /path/to/bazel-complete.bash` in your
   `~/.bashrc` file (under OS X, put it in your `~/.bash_profile` file).

### Getting zsh completion

Bazel also comes with a zsh completion script. To install it:

1. Add this script to a directory on your $fpath:

    ```
    fpath[1,0]=~/.zsh/completion/
    mkdir -p ~/.zsh/completion/
    cp scripts/zsh_completion/_bazel ~/.zsh/completion
    ```

2. Optionally, add the following to your .zshrc.

    ```
    # This way the completion script does not have to parse Bazel's options
    # repeatedly.  The directory in cache-path must be created manually.
    zstyle ':completion:*' use-cache on
    zstyle ':completion:*' cache-path ~/.zsh/cache
    ```

## Compiling from source

If you would like to build Bazel from source, clone the source from GitHub and
run `./compile.sh` to build it:

```
$ git clone https://github.com/bazelbuild/bazel.git
$ cd bazel
$ ./compile.sh
```

This will create a bazel binary in `bazel-bin/src/bazel`.

Check our [continuous integration](http://ci.bazel.io) for the current status of
the build.

For more information on using Bazel, see [Getting
started](getting-started.html).

---
layout: documentation
---

# Installing Bazel

## System Requirements

Supported platforms:

*   Ubuntu Linux
*   Mac OS X

Java:

*   Java JDK 7 or later

## Downloading Bazel

Clone the Bazel repo from GitHub:

```
$ cd $HOME
$ git clone https://github.com/google/bazel.git
```

## Building Bazel

### Building Bazel on Ubuntu

To build Bazel on Ubuntu:

#### 1. Install JDK 7:

**Ubuntu Utopic (14.10) and Trusty (14.04 LTS).** To install OpenJDK 7:

```
$ sudo apt-get install openjdk-7-jdk openjdk-7-source
```

#### 2. Install required packages:

```
$ sudo apt-get install pkg-config zip g++ zlib1g-dev
```

#### 3. Build Bazel:

```
$ cd bazel
$ ./compile.sh
```

If this fails to find a correct Java version, then try to
set the `JAVA_HOME` environment variable.

Find the Java `bin` directory using `readlink -f $(which javac)`
and use `javac -version` to verify that you have the right JDK version (1.7+).
Then set the `JAVA_HOME` environment variable to the `bin` directory
parent.

For example, if the path is `/usr/lib/jvm/jdk1.7.0/bin/javac`, set the
`JAVA_HOME` variable to `/usr/lib/jvm/jdk1.7.0`:

```
$ export JAVA_HOME=/usr/lib/jvm/jdk1.7.0
```

You can also add this line to your `~/.bashrc` file.


### Building Bazel on OS X

Bazel on Mac OS X requires:

* The Xcode command line tools. Xcode can be downloaded from the
  [Apple Developer Site](https://developer.apple.com/xcode/downloads/).

* MacPorts or Homebrew for installing required packages.

* An installation of JDK 7.

* For `objc_*` and `ios_*` rule support, you must have Xcode 6.1 or later with
  iOS SDK 8.1 installed on your system.

To build Bazel on Mac OS X:

```
$ cd bazel
$ ./compile.sh
```

Then you can run Bazel:

```
$ ./output/bazel help
```

## Running Bazel

The Bazel executable is located at `output/bazel` in the Bazel home directory.
It's a good idea to add this path to your default paths, as follows:

```bash
$ export PATH="$PATH:$HOME/bazel/output"
```

You can also add this command to your `~/.bashrc` file.

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

For more information, see [Getting started](getting-started.html).

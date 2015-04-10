---
layout: default
---

# Installing Bazel

## System Requirements

Supported platforms:

*   Ubuntu Linux
*   Mac OS X

Java:

*   Java JDK 8 or later

## Downloading Bazel

Clone the Bazel repo from GitHub:

```
$ cd $HOME
$ git clone https://github.com/google/bazel/
```

## Building Bazel

### Building Bazel on Ubuntu

To build Bazel on Ubuntu:

1.  Install JDK 8:

    *   **Ubuntu Trusty (14.04 LTS).** OpenJDK 8 is not available on Trusty. To
        install Oracle JDK 8:

        ```
        $ sudo add-apt-repository ppa:webupd8team/java
        $ sudo apt-get update
        $ sudo apt-get install oracle-java8-installer
        ```

    *   **Ubuntu Utopic (14.10).** To install OpenJDK 8:

        ```
        $ sudo apt-get install openjdk-8-jdk
        ```

2.  Set the `JAVA_HOME` environment variable.

    First, check to see if it's already set:

    ```
    $ echo $JAVA_HOME
    ```

    If this prints the path to the JDK 8 root directory, proceed to the next
    step. Otherwise, find the Java `bin` directory using `which javac` and use
    `javac -version` to verify that you have the right JDK version. Then set
    the `JAVA_HOME` environment variable to the `bin` directory parent.

    For example, if the path is `/usr/lib/jvm/jdk1.8.0/bin/javac`, set the
    `JAVA_HOME` variable to `/usr/lib/jvm/jdk1.8.0`:

    ```
    $ export JAVA_HOME=/usr/lib/jvm/jdk1.8.0
    ```

    You can also add this line to your `~/.bashrc` file.

3.  Install required packages:

    ```
    $ sudo apt-get install libarchive-dev pkg-config zip g++ zlib1g-dev
    ```

4.  Build Bazel:

    ```
    $ cd bazel
    $ ./compile.sh
    ```

### Building Bazel on OS X

Bazel on Mac OS X requires:

*   The Xcode command line tools. Xcode can be downloaded from the
    [Apple Developer Site](https://developer.apple.com/xcode/downloads/).

*   MacPorts or Homebrew for installing required packages.

*   An installation of JDK 8.

*   For `objc_*` and `ios_*` rule support, you must have Xcode 6.1 or later with
    iOS SDK 8.1 installed on your system.

To build Bazel on Mac OS X:

1.  Install the required packages:

    ```
    $ port install protobuf-cpp libarchive
    ```

    or

    ```
    $ brew install protobuf libarchive
    ```

2. Build Bazel:

   ```
   $ cd bazel
   $ ./compile.sh
   ```

3. Run Bazel:

   ```
   $ ./output/bazel help
   ```

## Running Bazel

The Bazel executable is located at `output/bazel` in the Bazel home directory.
It's a good idea to add this path to your default paths, as follows:

```
$ export PATH="$PATH:$HOME/bazel/output/bazel"
```

You can also add this command to your `~/.bashrc` file.

## Getting bash completion

Bazel come with a bash completion script. To install it:

1. Build it with Bazel: `bazel build //scripts:bazel-complete.bash`.
2. Copy the script `bazel-bin/scripts/bazel-complete.bash` to your
   completion folder (`/etc/bash_completion.d` directory under Ubuntu).
   If you don't have a completion folder, you can copy it wherever suits
   you and simply insert `source /path/to/bazel-complete.bash` in your
   `~/.bashrc` file (under OS X, put it in your ~/.bash_profile file).

For more information, see [Getting started](getting-started.html).

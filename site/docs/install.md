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

You must run the Bazel from within a directory that is properly configured for
use with the application. We call this directory a _workspace directory_.
Bazel provides a default workspace directory with sample `BUILD` files and
source code at `base_workspace` in the Bazel home directory. This directory
contains files and subdirectories that must be present in order for Bazel to
work. If you want to build from source outside the default workspace directory,
copy the entire `base_workspace` directory to the new location before adding
your `BUILD` and source files.

To run Bazel and build a sample Java application:

```
$ cp -R $HOME/bazel/base_workspace $HOME/my_workspace
$ cd $HOME/my_workspace
$ bazel build //examples/java-native/src/main/java/com/example/myproject:hello-world
```

The build output is located in
`$HOME/my_workspace/bazel-bin/examples/java-native/src/main/java/com/example/myproject/`.

To run the sample application:

```
$ $HOME/my_workspace/bazel-bin/examples/java-native/src/main/java/com/example/myproject/hello-world
```

For more information, see [Getting started](getting-started.md).

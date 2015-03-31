---
layout: default
---

# Installing Bazel

This file contains instructions for downloading, building and running Bazel.

## System Requirements

Supported platforms:

* Ubuntu Linux
* Mac OS X

Java:

* Java JDK 8 or later

## Getting Bazel

1. Clone the Bazel repo from GitHub:

        $ cd $HOME
        $ git clone https://github.com/google/bazel/

## Building Bazel

### Building Bazel on Ubuntu

To build Bazel on Ubuntu:

1. Install JDK 8:
  * Ubuntu Trusty (14.04LTS): OpenJDK 8 is not available so the
    fastest way is to install the Oracle JDK 8:

            $ sudo add-apt-repository ppa:webupd8team/java
            $ sudo apt-get update
            $ sudo apt-get install oracle-java8-installer
  * Ubuntu Utopic (14.10): you can install OpenJDK 8 like this:

            $ sudo apt-get install openjdk-8-jdk

2. Set the `JAVA_HOME` environment variable.

   Check if it's already set:

       $ echo $JAVA_HOME

   If this prints the path to the JDK 8 root, you can proceed to the next step.

   Otherwise you'll need to set this to the root of the JDK. Use `which javac`
   to find the path to the JDK's `bin` directory. Use `javac -version` to verify
   that you're dealing with the right JDK version.

   For example the path could be `/usr/lib/jvm/jdk1.8.0/bin/javac`. Set the
   `JAVA_HOME` variable to `/usr/lib/jvm/jdk1.8.0` then, using:

       $ export JAVA_HOME=/usr/lib/jvm/jdk1.8.0

   You can also add this line to your `~/.bashrc` file.

3. Install required packages:

        $ sudo apt-get install libarchive-dev pkg-config zip g++ zlib1g-dev

4. Build Bazel:

        $ cd bazel
        $ ./compile.sh


### Building Bazel on OS X

Using Bazel on Mac OS X requires:

* The Xcode command line tools. Xcode can be downloaded from
  [Apple's developer site](https://developer.apple.com/xcode/downloads/).
* MacPorts or Homebrew for installing required packages.
* An installation of JDK 8.
* (optional) For `objc_*` and `ios_*` rule support: An installed copy of
  Xcode 6.1 or later with iOS SDK 8.1.

To build Bazel on Mac OS X:


1. Install required packages:

        $ port install protobuf-cpp libarchive

   or

        $ brew install protobuf libarchive

2. Build Bazel:

        $ cd bazel
        $ ./compile.sh

3. Run it

        $ ./output/bazel help

## Running Bazel


The Bazel executable is located at `output/bazel` in the bazel directory.
It's a good idea to add this path to your default paths, like so (you
can also add this command to your `~/.bashrc`):

    $ export PATH="$PATH:$HOME/bazel/output/bazel"

You must run Bazel from within a source code tree that is properly
configured for use with Bazel. We call such a tree a _workspace
directory_. Bazel provides a default workspace directory with sample
`BUILD` files and source code in `base_workspace`. The default
workspace contains files and directories that must be present in order
for Bazel to work. If you want to build from source outside the
default workspace directory, copy the entire `base_workspace`
directory to the new location before adding your `BUILD` and source
files.

Build a sample Java application:

        $ cp -R $HOME/bazel/base_workspace $HOME/my_workspace
        $ cd $HOME/my_workspace
        $ bazel build //examples/java-native/src/main/java/com/example/myproject:hello-world

The build output is located in `$HOME/my_workspace/bazel-bin/examples/java-native/src/main/java/com/example/myproject/`.

Run the sample application:

    $ $HOME/my_workspace/bazel-bin/examples/java-native/src/main/java/com/example/myproject/hello-world

For more information, see [Getting started](getting-started.html).

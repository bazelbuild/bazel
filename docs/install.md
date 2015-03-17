# Compiling Bazel

This file contains instructions for building and running Bazel.

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

1. Install required package:

        $ sudo apt-get install libarchive-dev openjdk-8-jdk

2. Build Bazel:

        $ cd bazel
        $ ./compile.sh

### Building Bazel on OS X

Using Bazel on Mac OS X requires:

* The Xcode command line tools. Xcode can be downloaded from
  [Apbple's developer site](https://developer.apple.com/xcode/downloads/).
* MacPorts or Homebrew for installing required packages.
* An installation of JDK 8.

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

The Bazel executable is located at `output/bazel`, under the source
tree root.

You must run Bazel from within source code tree, which is properly
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
        $ $HOME/bazel/output/bazel build //examples/java-native/src/main/java/com/example/myproject:hello-world

The build output is located in `$HOME/my_workspace/bazel-bin/examples/java-native/src/main/java/com/example/myproject/`.

Run the sample application:

    $ $HOME/my_workspace/bazel-bin/examples/java-native/src/main/java/com/example/myproject/hello-world

For more information, see [Getting started](getting-started.md).

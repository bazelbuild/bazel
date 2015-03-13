*Bazel is very much a work in progress. We'd love if you tried it out, but there
are many rough edges. Please feel free to
[give us feedback](https://groups.google.com/forum/#!forum/bazel-discuss)!*

# Bazel

*{Fast, Correct} - Choose two*

Bazel is an build tool that builds code quickly and reliably.
It executes as few build steps as possible by tracking dependencies and outputs,
controls the build environment to keep builds hermetic, and uses its
knowledge of dependencies to parallelize builds.

This README file contains instructions for building and running Bazel.

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

### Building Bazel on OS X (experimental)

Using Bazel on Mac OS X requires:

* Xcode and Xcode command line tools
* MacPorts or Homebrew for installing required packages
* A JDK 8 installed

To build Bazel on Mac OS X:

1. Install required packages:

        $ port install protobuf-cpp libarchive

   or

        $ brew install protobuf libarchive

2. Build Bazel:

        $ cd bazel
        $ ./compile.sh

## Running Bazel

The Bazel executable is located at `<bazel_home>/output/bazel`.

You must run Bazel from within a _workspace directory_. Bazel provides a default
workspace directory with sample `BUILD` files and source code in
`<bazel_home>/base_workspace`. The default workspace contains files and
directories that must be present in order for Bazel to work. If you want to
build from source outside the default workspace directory, copy the entire
`base_workspace` directory to the new location before adding your `BUILD` and
source files.

Build a sample Java application:

        $ cp -R $HOME/bazel/base_workspace $HOME/my_workspace
        $ cd $HOME/my_workspace
        $ $HOME/bazel/output/bazel build //examples/java-native/src/main/java/com/example/myproject:hello-world

The build output is located in `$HOME/my_workspace/bazel-bin/examples/java-native/src/main/java/com/example/myproject/`.

Run the sample application:

    $ $HOME/my_workspace/bazel-bin/examples/java-native/src/main/java/com/example/myproject/hello-world

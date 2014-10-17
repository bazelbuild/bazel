Building Bazel
==============

We currently only support building on Ubuntu, and the binaries only run on
Ubuntu. You will need packages for the protobuf-compiler, and for libarchive:

    sudo apt-get install protobuf-compiler libarchive-dev

You will also need a JDK installed. Then run:

    ./compile.sh

This will create the bazel binary in the output directory. We are working on
bootstrapping bazel with itself.

Experimental: Building Bazel on OS X
------------------------------------

You will need Xcode, the Xcode command line tools, and a JDK installed. Bazel's
compile script assumes you're using MacPorts or Homebrew to install
dependencies.  To install the prerequisites, run:

    port install protobuf-cpp libarchive

or

    brew install protobuf libarchive

Once the prerequisites are installed, follow the building instructions above.

Running Bazel
=============

Bazel requires certain files to exist in certain places. To get your workspace
set up correctly, start by copying `base_workspace/` to wherever you want to do
your builds and make your project a subdirectory of `base_workspace/`.

Create a BUILD file in a subdirectory of `base_workspace`, for example:

    $ cd base_workspace
    $ mkdir hello
    $ echo 'genrule(name = "world", outs = ["hi"], cmd = "touch $@")' > hello/BUILD

Now run Bazel, e.g.,

    $ bazel build //hello:world

You should be able to find the file "hi" in bazel-genfiles/hello/.

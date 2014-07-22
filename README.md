Building Bazel
==============

We currently only support building on Ubuntu, and the binaries only run on
Ubuntu. You will need packages for the protobuf-compiler, and for libarchive:

    apt-get install protobuf-compiler libarchive-dev

Then run:

    ./compile.sh

This will create the bazel binary in the output directory. We are working on
bootstrapping bazel with itself.

Running Bazel
=============

Bazel requires certain files to exist in certain places. To get your workspace
set up correctly, copy `example_workspace/` to wherever you want to do your
builds.

Create your own project with a BUILD file in a subdirectory, for example:

    $ cd example_workspace
    $ mkdir hello
    $ echo 'genrule(name = "world", outs = ["hi"], cmd = "touch $(@D)/hi")' > hello/BUILD

Now run Bazel, e.g.,

    $ bazel build //hello:world

You should be able to find the file "hi" in blaze-genfiles/hello/.

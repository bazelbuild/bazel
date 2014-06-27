Building Bazel
==============

We currently only support building on Ubuntu, and the binaries only run on
Ubuntu. You will need packages for the protobuf-compiler, and for libarchive:

    apt-get install protobuf-compiler libarchive-dev

Then run:

    ./compile.sh

We are working on bootstrapping bazel with itself.

Running Bazel
=============

Bazel has some workspace setup requirements (which we're working on removing).
All builds need to happen from within a google3 directory (or subdirectory) and
have certain files exist in certain places. To get your workspace set up
correctly, copy example-workspace/google3 to wherever you want to do your
builds.

Create your own project with a BUILD file within this google3 directory, for
example:

    $ cd google3
    $ mkdir -p hello
    $ echo 'genrule(name = "world", outs = ["hi"], cmd = "touch $(@D)/hi")' > hello/BUILD

Now run Bazel, e.g.,

    $ bazel build //hello:world

You should be able to find the file "hi" in blaze-genfiles/hello/.

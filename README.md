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

Create a google3 directory.  This is your package root where all of your builds
will happen (we're working on removing the google3 naming requirement).  Add the
following directory structure:

    $ cd google3
    $ mkdir -p tools/genrule my_output my_install
    $ touch tools/genrule/genrule-setup.sh

Create tools/genrule/BUILD and add the following to it:

    exports_files([
        "genrule-setup.sh",
    ])

Create your own project with a BUILD file, for example:

    $ mkdir -p hello
    $ echo 'genrule(name = "world", outs = ["hi"], cmd = "touch $(@D)/hi"' > hello/BUILD
)

Now run Bazel using --output_base and --install_base options, e.g.,

    $ bazel --output_base=my_output --install_base=my_install build //hello:world

You should be able to find the file "hi" in blaze-genfiles/hello/.

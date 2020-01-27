To reproduce this tree, to:

1. `git clone https://github.com/nanopb/nanopb.git` in a convenient directory
2. `git checkout <tag>` (current is `0.3.6`, commithash  `eb0e73ca`)
   Note that we want the same nanopb release used by gRPC upstream for the
   version in third_party/grpc.
3. `rm -fr <nanopb git tree>/tests`
4. `cp -R <nanopb git tree>/* third_party/nanopb
5. Update BUILD and README.bazel.md if necessary.

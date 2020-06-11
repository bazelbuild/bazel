# How to update the C++ sources of gRPC:

1. Update the gRPC definitions in WORKSPACE file, currently we use 
   https://github.com/grpc/grpc/archive/v1.26.0.tar.gz
2. Update the gRPC patch file if necessary, it mostly helps avoid unnecessary dependencies.
3. Update third_party/grpc/BUILD to redirect targets to @com_github_grpc_grpc if necessary.

# How to update the BUILD/bzl sources of gRPC:

1. `git clone http://github.com/grpc/grpc.git` in a convenient directory
2. `git checkout <tag>` (current is `v1.26.0`, commithash `de893acb`)
3. `mkdir -p third_party/grpc/bazel`
4. `cp <gRPC git tree>/bazel/{BUILD,cc_grpc_library.bzl,generate_cc.bzl,protobuf.bzl} third_party/grpc/bazel`
5. In the `third_party/grpc` directory, apply local patches:
   `patch -p3 < bazel.patch`

# How to update the Java plugin:

1. Checkout tag `v1.26.0` from https://github.com/grpc/grpc-java
2. `cp -R <grpc-java git tree>/compiler/src/java_plugin third_party/grpc/compiler/src`

# How to update the Java code:

Download the necessary jars at version `1.26.0` from maven central.

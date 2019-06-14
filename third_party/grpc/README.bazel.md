# How to update the C++ sources of gRPC:

1. `git clone http://github.com/grpc/grpc.git` in a convenient directory
2. `git checkout <tag>` (current is `v1.18.0`, commithash `007b721f`)
3. `mkdir -p third_party/grpc/src`
4. `cp -R <gRPC git tree>/src/{compiler,core,cpp} third_party/grpc/src`
5. `cp -R <gRPC git tree>/include third_party/grpc`
6. `rm -rf third_party/grpc/src/core/tsi/test_creds`
7. Update BUILD files by copying the rules from the BUILD file of gRPC;
   fix macros in third_party/grpc/build_defs.bzl if necessary
8. Apply local patches if necessary: `patch -p3 < netinet_tcp_h.patch`
9. Update //third_party/nanopb if necessary


# How to update the Java plugin:

1. Checkout tag `v1.10.0` from https://github.com/grpc/grpc-java
2. `cp -R <grpc-java git tree>/compiler/src/java_plugin third_party/grpc/compiler/src`

# How to update the Java code:

Download the necessary jars at version `1.20.0` from maven central.

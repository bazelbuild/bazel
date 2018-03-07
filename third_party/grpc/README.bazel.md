# How to update the C++ sources of gRPC:

1. `git clone http://github.com/grpc/grpc.git` in a convenient directory
2. `git checkout <tag>` (current is `release-0_13`, commithash `78e04bbd`)
3. `mkdir -p third_party/grpc/src`
4. `cp -R <gRPC git tree>/src/{compiler,core-cpp} third_party/grpc/src`
5. `cp -R <gRPC git tree>/include third_party/grpc`
6. Update BUILD files by copying the rules from the BUILD file of gRPC
7. Patch in grpc.patch. It makes gRPC work under msys2.


# How to update the Java plugin:

1. Checkout tag `v1.10.0` from https://github.com/grpc/grpc-java
2. `cp -R <grpc-java git tree>/compiler/src/java_plugin third_party/grpc/compiler/src`

# How to update the Java code:

Download the necessary jars at version `1.10.0` from maven central.

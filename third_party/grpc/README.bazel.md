How to update the C++ sources of gRPC:

1. `git clone http://github.com/grpc/grpc.git` in a convenient directory
2. `git checkout <tag>` (current is `release-0_13`, commithash `78e04bbd`)
3. `mkdir -p third_party/grpc/src`
4. `cp -R <gRPC git tree>/src/{compiler,core-cpp} third_party/grpc/src`
5. `cp -R <gRPC git tree>/include third_party/grpc`
6. Update BUILD files by copying the rules from the BUILD file of gRPC
7. Patch in grpc.patch. It makes gRPC work under msys2.


How to update the Java plugin:

Download it from Maven central. The project is called `protoc-gen-grpc-java`
and the version is `0.15.0` .

How to update the Java code:

Download it from Maven central. The jars are called `grpc-core`, `grpc-netty`,
`grpc-protobuf`, `grpc-protobuf-lite`, `grpc-stub` and the version is
`0.15.0`.

* * *
How to update the Linux s390x 64-bit Java plugin:

1. Build Java plugin version `0.15.0` from https://github.com/grpc/grpc-java .
2. `cp <gRPC git tree>/compiler/build/artifacts/java_plugin/protoc-gen-grpc-java.exe <Bazel tree>/third_party/grpc/protoc-gen-grpc-java-0.15.0-linux-s390x_64.exe`

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
and the version is `0.14.1` .

How to update the Java code:

Download it from Maven central. The jars are called `grpc-core`, `grpc-netty`,
`grpc-protobuf`, `grpc-protobuf-lite`, `grpc-stub` and the version is
`0.14.1`.

How to update the C++ sources of gRPC:

1. `git clone http://github.com/grpc/grpc.git` in a convenient directory
2. `git checkout <tag>` (current is `release-0_13`, commithash `78e04bbd`)
3. `mkdir -p third_party/grpc/src`
4. `cp -R <gRPC git tree>/src/{compiler,core-cpp} third_party/grpc/src`
5. `cp -R <gRPC git tree>/include third_party/grpc`
6. Update BUILD files by copying the rules from the BUILD file of gRPC
7. Patch in grpc.patch. It makes gRPC work under msys2.


How to update the Java plugin:

For any architecture other than 64-bit Linux, downloading it from Maven Central
is fine. For 64-bit Linux, the plugin needs to be built statically:

1. `git clone http://github.com/grpc/grpc-java`
2. `git checkout <tag>` (current is `v0.13.2`, commithash `5933cea9`)
3. Modify the linker args in compiler/build.gradle according to the instructions below
4. Download the sources of protobuf (see `third_party/protobuf/README.md`) and compile it
5. `export LDFLAGS=<protobuf dir>/src/.libs`
6. `export CXXFLAGS=<protobuf dir>/src`
7. `cd compiler; ../gradlew java_pluginExecutable`

In `compiler/build.gradle`, this list of linker arguments:

```
          linker.args "-Wl,-Bstatic", "-lprotoc", "-lprotobuf", "-static-libgcc",
                      "-static-libstdc++",
                      "-Wl,-Bdynamic", "-lpthread", "-s"
```

needs to be replaced with this:

```
          linker.args "-Wl,-Bstatic", "-lprotoc", "-lprotobuf", "-static-libgcc",
                      "-static-libstdc++",
                      "-lpthread", "-lc", "-s", "-static"
```

How to update the Java code:

Simply download from Maven Central.

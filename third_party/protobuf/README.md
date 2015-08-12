How to update these files:

1. Go to http://search.maven.org/
2. Search for g:"com.google.protobuf"
3. Download the "jar" link from protobuf-java.
4. Download all binaries from "protoc".
5. Strip version number from protoc files: for i in *.exe; do mv $i $(echo $i | sed s/3.0.0-alpha-3-//); done
6. Set executable bit: chmod +x *.exe
7. Update third_party/BUILD to point to the new jar file.
8. Done.


Because maven.org doesn't have a prebuilt binary for linux on ARM, you need to build the binary
yourself on the target system. Follow the build steps of protocol buffer to create the binary,
copy it to this directory and rename it to "protoc-linux-arm32.exe".

For example:

$ cp /usr/bin/protoc $BAZEL/third_party/protobuf/protoc-linux-arm32.exe

This should be done before you run ./compile.sh.


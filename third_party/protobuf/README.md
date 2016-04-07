How to update the binaries other than `protoc-linux-x86_64.exe` and `protoc-linux-arm32.exe`:

1. Go to http://search.maven.org/
2. Search for g:"com.google.protobuf"
3. Download the "jar" link from protobuf-java.
4. Download all binaries from "protoc".
5. Strip version number from protoc files: for `i in *.exe; do mv $i $(echo $i | sed s/3.0.0-beta-2-//); done`
6. Set executable bit: `chmod +x *.exe`
7. Update third_party/BUILD to point to the new jar file.
8. Done.

The 64-bit Linux version of the proto compiler is linked statically. To update it, do
the following steps on an x86_64 machine:

1. `git clone http://github.com/google/protobuf.git`
2. `git checkout <commithash>` (current is `d5fb408d` or `3.0.0-beta-2`)
3. `./autogen.sh`
4. `LDFLAGS=-static ./configure`
5. Change `LDFLAGS = -static` to `LDFLAGS = -all-static` in  `src/Makefile`.
6. `make`
7. `cp src/protoc <Bazel tree>/third_party/protobuf/protoc-linux-x86_64.exe` .

How to update the `src/` directory:

1. Run `git clone http://github.com/google/protobuf.git` in a convenient directory.
2. `mkdir third_party/protobuf/src/google` in the root of the Bazel tree.
3. `cp -R <root of protobuf tree>/src/google/protobuf third_party/protobuf/src/google`
4. Done.

The current version comes from commit `698fa8ee22`.

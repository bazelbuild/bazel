How to update the binaries other than `protoc-linux-x86_64.exe` and `protoc-mingw.exe`:

1. Go to http://search.maven.org/
2. Search for g:"com.google.protobuf"
3. Download the "jar" link from protobuf-java and put them in `<Bazel tree>/third_party/protobuf/<version>`
4. Download all binaries from "protoc" and put them in `<Bazel tree>/third_party/protobuf/<version>`
5. Set executable bit: `chmod +x *.exe`

The 64-bit Linux version of the proto compiler (`protoc-linux-x86_64.exe`) is
linked statically. To update it, do the following steps on an x86_64 machine:

1. `git clone http://github.com/google/protobuf.git`
2. `git checkout <commithash>`
3. `./autogen.sh`
4. `LDFLAGS=-static ./configure`
5. Change `LDFLAGS = -static` to `LDFLAGS = -all-static` in  `src/Makefile`.
6. `make`
7. `cp src/protoc <Bazel tree>/third_party/protobuf/<version>/protoc-<version>-linux-x86_64.exe` .

How to update `protoc-mingw.exe`:
This is pretty much the same steps as for x86_64 above, but they need to be done
from MingW shell on a Windows machine ([https://sourceforge.net/projects/msys2/files/]).

1. `git clone http://github.com/google/protobuf.git`
2. `git checkout <commithash>`
3. `./autogen.sh`
6. `make`
7. `cp src/protoc.exe <Bazel tree>/third_party/protobuf/<version>/protoc-<version>-mingw.exe` .

How to update `protobuf.bzl` and the `src/` directory:

1. `git clone http://github.com/google/protobuf.git`
2. `git checkout <commithash>`
2. `mkdir -p third_party/protobuf/<version>/src/google` in the root of the Bazel tree.
3. `cp -R <root of protobuf tree>/src/google/protobuf third_party/protobuf/<version>/src/google`

Finally, update the rules:

1. Add a BUILD file to `third_party/protobuf/<version>/`. Use the BUILD file
   for the previous version as a template. Update the `cc_library` rules to
   match the rules in the BUILD file in the protobuf repository. Also copy
   `protobuf.bzl` from the protobuf repository into
   `third_party/protobuf/<version>/`.
2. Modify `third_party/protobuf/BUILD` to point to the new rules.
3. Delete the old version of protobuf.

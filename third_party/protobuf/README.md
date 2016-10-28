### Updating binaries other than the Linux 64-bit and MinGW ones

1. Go to http://search.maven.org/
2. Search for g:"com.google.protobuf"
3. Download the "jar" link from protobuf-java and put them in `<Bazel tree>/third_party/protobuf/<version>`
4. Download all binaries from "protoc" and put them in `<Bazel tree>/third_party/protobuf/<version>`
5. Set executable bit: `chmod +x *.exe`

* * *
### Updating the Linux 64-bit proto compiler
The 64-bit Linux version of the proto compiler is linked statically. To update it, do
the following steps on an x86_64 machine:

1. `git clone http://github.com/google/protobuf.git`
2. `git checkout <tag or commithash>` (e.g. `v3.0.0` or `e8ae137`)
3. `./autogen.sh`
4. `LDFLAGS=-static ./configure`
5. Change `LDFLAGS = -static` to `LDFLAGS = -all-static` in  `src/Makefile`.
6. `make`
7. `cp src/protoc <Bazel tree>/third_party/protobuf/<version>/protoc-<version>-linux-x86_64.exe` .

* * *
### Updating the MinGW proto compiler (64-bit)
Do this from a MinGW shell ([https://sourceforge.net/projects/msys2/files/]) on
a Windows machine.

1. Clone the protobuf repo and check out the right commit

   ```sh
   git clone http://github.com/google/protobuf.git
   git checkout <tag or commithash>   # e.g. `v3.0.0` or `e8ae137`
   ```

2. Close all other MSYS/MinGW/Cygwin windows. Kill all running background jobs.
   This step is optional, but if you have other terminal windows open the next
   step may print some error messages (though it still seems to work).

3. Install necessary MinGW packages

   ```sh
   pacman -Syuu autoconf automake libtool curl make gcc unzip
   ```

4. Configure for static linking and build like you would for Unix

   ```sh
   ./autogen.sh
   ./configure --disable-shared   # takes about 2 minutes
   ./make                         # takes about 11 minutes
   ```

5. Copy resulting binary

   ```sh
   cp src/protoc.exe <bazel tree>/third_party/protobuf/protoc-mingw.exe
   ```

* * *
### Updating the Linux s390x 64-bit proto compiler
To add 64-bit Linux s390x version of the statically linked proto compiler, use below steps:

1. Build Protobuf compiler (v3.0.0-beta-2) from https://github.com/google/protobuf.
2. `cp src/protoc <Bazel tree>/third_party/protobuf/protoc-linux-s390x_64.exe`
3. `cp src/protoc <Bazel tree>/third_party/protobuf/<version>/protoc-linux-s390x_64.exe`


* * *
### Updating `protobuf.bzl` and the `src/` directory:

1. `git clone http://github.com/google/protobuf.git`
2. `git checkout <tag or commithash>` (e.g. `v3.0.0` or `e8ae137`)
3. `mkdir -p third_party/protobuf/<version>/src/google` in the root of the Bazel tree.
4. `cp -R <root of protobuf tree>/src/google/protobuf third_party/protobuf/src/google`
5. Update the rules in `third_party/protobuf/BUILD` with the rules in the protobuf repository.

Finally, update the rules:

1. Add a BUILD file to `third_party/protobuf/<version>/`. Use the BUILD file
   for the previous version as a template. Update the `cc_library` rules to
   match the rules in the BUILD file in the protobuf repository. Also copy
   `protobuf.bzl` from the protobuf repository into
   `third_party/protobuf/<version>/`.
2. Modify `third_party/protobuf/BUILD` to point to the new rules.
3. Delete the old version of protobuf.

* * *
### Updating anything else in the directory
Follow usual procedure as described on https://www.bazel.build/contributing.html

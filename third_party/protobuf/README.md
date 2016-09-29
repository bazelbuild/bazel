### Updating binaries other than the Linux 64-bit and MinGW ones

1. Go to http://search.maven.org/
2. Search for g:"com.google.protobuf"
3. Download the "jar" link from protobuf-java.
4. Download all binaries from "protoc".
5. Strip version number from protoc files: for `i in *.exe; do mv $i $(echo $i | sed s/3.0.0-beta-2-//); done`
6. Set executable bit: `chmod +x *.exe`
7. Update `third_party/BUILD` to point to the new jar file.
8. Done.

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
7. `cp src/protoc <Bazel tree>/third_party/protobuf/protoc-linux-x86_64.exe` .

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
### Updating `protobuf.bzl` and the `src/` directory:

1. `git clone http://github.com/google/protobuf.git`
2. `git checkout <tag or commithash>` (e.g. `v3.0.0` or `e8ae137`)
2. `mkdir -p third_party/protobuf/src/google` in the root of the Bazel tree.
3. `cp -R <root of protobuf tree>/src/google/protobuf third_party/protobuf/src/google`
4. Update rules in `third_party/protobuf/BUILD` with the rules in the protobuf repository.
5. Done.

* * *
### Updating anything else in the directory
Follow usual procedure as described on https://www.bazel.io/contributing.html

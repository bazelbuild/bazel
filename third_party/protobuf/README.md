### Updating the jar binary

1. Go to http://search.maven.org/
2. Search for g:"com.google.protobuf"
3. Download the "jar" link from protobuf-java and put them in `<Bazel tree>/third_party/protobuf/<version>`

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

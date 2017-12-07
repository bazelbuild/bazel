# Updating protobuf


1) Fetch the desired protobuf version and copy it in a folder `new_proto` under
`third_party/protobuf`.
2) Bazel uses upstream protobuf from source, except for java, as we currently don't
build protobuf java when bootstrapping bazel. So instead we use pre-built jars.
So build the java proto library from source and in case you cloned an upstream version
of protobuf, remove the .git folders:
```
cd new_proto
bazel build :protobuf_java :protobuf_java_util
cp bazel-bin/libprotobuf_java.jar .
cp bazel-bin/libprotobuf_java_util.jar .
bazel clean --expunge
rm -rf .git .gitignore .gitmodules
```
3) Modify protobuf's `BUILD` file to not build java from source, but to use
   the jars instead. To do that, in the BUILD file delete the rules listed
   under `Java support`. Then, from the `third_party/protobuf/<old_proto>/BUILD file`
   copy the rules under "Modifications made by bazel" to the new BUILD file.
   The java rules in there should have the same names as the ones you just deleted under "Java support".
   You might need to update the names of the jars in the rules sources to the ones you just build.
4) Copy `third_party/protobuf/<old_proto>/com_google_protobuf_java.BUILD` to the new
   directory.
5) Copy the `licenses` declaration and the `srcs` filegroup from
   `third_party/protobuf/<old_proto>/util/python/BUILD` to the corresponding
   file in the new directory.
6) For each file listed in `RELATIVE_WELL_KNOWN_PROTOS` in the `new_proto/BUILD` file
   copy it from `new_proto/srcs/google/protobuf` to `new_proto/google/protobuf`.
7) Name the `new\_proto` directory according to the protobuf version number.
8) In `third\_party/protobuf/BUILD` update `PROTOBUF\_VERSION` to the name of the
directory you just created.
9) In the root `WORKSPACE` file update relative paths of protobuf to point to
the new version.
10) Delete the `third_party/protobuf/<old_proto>` directory.
11) Update this file if you found the :instructions to be wrong or incomplete.

# Current protobuf version

The current version of protobuf is [3.4.0](https://github.com/google/protobuf/releases/tag/v3.4.0).

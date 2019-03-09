# Updating protobuf

You will create and merge the following Pull Requests.

## 1st PR: add the new protobuf version to the Bazel tree

1.  Fetch the desired protobuf version and copy it in a folder `new_proto` under
   `third_party/protobuf`.

    **Example:** to upgrade to 3.7.1, download and unpack
    [protobuf-all-3.7.1.zip](https://github.com/protocolbuffers/protobuf/releases/download/v3.7.1/protobuf-all-3.7.1.zip).

1.  Build the Java proto library from source and, in case you cloned an upstream version
    of protobuf, remove the .git folders:

    ```
    cd third_party/protobuf/new_proto
    bazel build :protobuf_java :protobuf_java_util
    cp bazel-bin/libprotobuf_java.jar .
    cp bazel-bin/libprotobuf_java_util.jar .
    bazel clean --expunge
    rm -rf .git .gitignore .gitmodules
    ```

    **Reason:** Bazel uses upstream protobuf from source, except for Java, as Bazel's bootstrapping
    scripts currently don't build protobuf Java when bootstrapping Bazel but use pre-built jars
    instead.

1.  Modify protobuf's `BUILD` file to not build java from source, but to use the jars instead:

    1.  In the BUILD file delete the rules listed under `Java support`.

    1.  From the `third_party/protobuf/<old_proto>/BUILD` file copy the rules under the
        "Modifications made by bazel" section to the new BUILD file. The java rules in there should
        have the same names as the ones you just deleted under "Java support". You might need to
        update the names of the jars in the rules sources to the ones you just build.

1.  Copy `third_party/protobuf/<old_proto>/com_google_protobuf_java.BUILD` to the new directory.

1.  From `third_party/protobuf/<old_proto>/util/python/BUILD`,
    `third_party/protobuf/<old_proto>/examples/BUILD`, and
    `third_party/protobuf/<old_proto>/third_party/googletest/BUILD.bazel`:
    copy the `licenses` declaration and the `srcs` filegroup to the corresponding file under
    `third_party/protobuf/<new_proto>`.

1.  In `third_party/protobuf/<new_proto>/BUILD`, replace zlib dependency from:
    ZLIB_DEPS = ["//external/zlib"]
    to
    ZLIB_DEPS = ["@//third_party/zlib"]

1.  In `third_party/protobuf/<new_proto>/BUILD`, in the `srcs` filegroup rule, update the version
    number referring to the newly added `srcs` rules.

1.  Rename `third_party/protobuf/<new_proto>` directory according to the protobuf version number.

1.  In `third_party/protobuf/BUILD`:

    1.  Add a new variable `_NEW_PROTOBUF_VERSION`, set to value of the version.

    1.  In the `srcs` filegroup rule, add:

        ```diff
         srcs = [
             "//third_party/protobuf/" + PROTOBUF_VERSION + ":srcs",
        +    "//third_party/protobuf/" + _NEW_PROTOBUF_VERSION + ":srcs",
         ],
        ```

1.  Create a PR of these changes and merge it directly to
    https://bazel.googlesource.com/bazel/+/master (without the usual process of importing it to
    the Google-internal version control).

## 2nd and 3rd PRs: update references in the Bazel tree

1.  In `third_party/protobuf/BUILD`:

    1.  rename `PROTOBUF_VERSION` to `_OLD_PROTOBUF_VERSION`

    1.  rename `_NEW_PROTOBUF_VERSION` to `PROTOBUF_VERSION`

1.  In the root `WORKSPACE` file update relative paths of protobuf to point to the new version.

1.  Update version number in `src/main/protobuf/BUILD` and `src/test/shell/testenv.sh`.

1.  Update the current version in this file.

1.  Create a PR of these changes and get it imported. Some files won't be imported (those that are
    only hosted on GitHub); this is expected.

2.  Wait for the imported PR to be pushed back to GitHub. Rebase the PR from the previous step, and
    merge it to https://bazel.googlesource.com/bazel/+/master .

## 4th PR: remove the old directory

1.  Delete the `third_party/protobuf/<old_proto>` directory.

1.  Remove `_OLD_PROTOBUF_VERSION` from `third_party/protobuf/BUILD`.

1.  Create a PR of these changes and merge it directly to
    https://bazel.googlesource.com/bazel/+/master .

**Update this file if you found these instructions to be wrong or incomplete.**

# Current protobuf version

The current version of protobuf is [3.7.1](https://github.com/google/protobuf/releases/tag/v3.7.1).

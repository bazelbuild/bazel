# Build definitions for Protolark-based Bazel configuration files (PROJECT.scl).
#
# See https://github.com/bazelbuild/bazel/issues/24839.

package(
    default_visibility = ["//visibility:private"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "project_proto_scl_filegroup",
    srcs = ["project_proto.scl"],
    visibility = ["//src/test/java/com/google/devtools/build/lib/analysis/util:__pkg__"],
)

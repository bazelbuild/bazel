# Build file to expose zipper to the @bazel_tools repository.
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

# zipper will be added when creating the @bazel_tools repository.
filegroup(
    name = "zipper",
    srcs = glob(["zipper/*"]),
)

sh_binary(
    name = "unzip",
    srcs = ["unzip.sh"],
)

sh_binary(
    name = "zip_manifest_creator",
    srcs = ["zip_manifest_creator.sh"],
)

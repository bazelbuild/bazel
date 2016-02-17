# Build file to expose zipper to the @bazel_tools repository.
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = ["BUILD"],
)

# zipper will be added when creating the @bazel_tools repository.
filegroup(
    name = "zipper",
    srcs = glob(["zipper/*"]),
)

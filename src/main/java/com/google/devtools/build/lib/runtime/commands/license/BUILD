load(":merge_licenses.bzl", "merge_licenses")

package(default_visibility = ["//src:__subpackages__"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

# Satisfy the rules in runtime/commands/BUILD which use a generated LICENSE file in Bazel open source
merge_licenses(
    name = "license",
    srcs = [
        "//:LICENSE",
        "//third_party:srcs",
    ],
    out = "LICENSE",
)

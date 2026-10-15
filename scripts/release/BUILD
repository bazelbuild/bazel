# Scripts for building Bazel releases
package(default_visibility = ["//visibility:private"])

exports_files(
    ["common.sh"],
    visibility = ["//scripts/docs:__pkg__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//scripts:__pkg__"],
)

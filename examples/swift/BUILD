package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("//tools/build_defs/apple:swift.bzl", "swift_library")

swift_library(
    name = "swift_lib",
    srcs = glob(["*.swift"]),
    deps = ["//examples/swift/BarLib"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//examples/swift/BarLib:srcs",
    ],
    visibility = ["//examples:__pkg__"],
)

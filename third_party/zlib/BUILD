licenses(["notice"])  #  BSD/MIT-like license

exports_files(["BUILD.tools"])

filegroup(
    name = "build_files",
    srcs = [
        "BUILD.tools",
        "LICENSE.txt",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//third_party:__pkg__"],
)

alias(
    name = "zlib",
    actual = "@zlib//:zlib",
    visibility = ["//visibility:public"],
)

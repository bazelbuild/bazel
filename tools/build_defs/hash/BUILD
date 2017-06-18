licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools:__pkg__"],
)

exports_files(
    ["README.md"],
    visibility = ["//site:__pkg__"],
)

py_binary(
    name = "sha256",
    srcs = ["sha256.py"],
    visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files(
    [
        "docs/user-manual.md",
        "versions/_toc.yaml",
    ],
    visibility = [
        "//src/test/java/com/google/devtools/build/lib/packages:__pkg__",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:private"],
)

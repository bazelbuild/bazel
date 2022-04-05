package(default_applicable_licenses = ["//third_party/devsite/bazel:license"])

licenses(["notice"])

exports_files(
    ["docs/user-manual.md"],
    visibility = [
        "//src/test/java/com/google/devtools/build/lib/packages:__pkg__",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//:__pkg__"],
)

load("@rules_pkg//pkg:pkg.bzl", "pkg_tar")

licenses(["notice"])

exports_files(
    [
        "docs/user-manual.md",
        "versions/_toc.yaml",
    ],
    visibility = [
        "//scripts/docs:__pkg__",
        "//src/test/java/com/google/devtools/build/lib/packages:__pkg__",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//scripts/docs:__pkg__",
        "//site:__pkg__",
    ],
)

pkg_tar(
    name = "docs",
    srcs = glob(
        ["**"],
        exclude = ["**/BUILD"],
    ),
    strip_prefix = "/site/en",
    visibility = ["//scripts/docs:__pkg__"],
)

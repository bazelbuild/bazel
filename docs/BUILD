# The Bazel docs
load("@rules_pkg//pkg:pkg.bzl", "pkg_tar")

package(default_applicable_licenses = ["//:license"])

licenses(["notice"])

exports_files(
    [
        "docs/user-manual.md",
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
        "//:__pkg__",  # Scheuklappen: keep
        "//scripts/docs:__pkg__",  # Scheuklappen: keep
    ],
)

# Snapshot of the full site, including stylesheets and images.
pkg_tar(
    name = "docs",
    srcs = glob(
        ["**"],
        exclude = [
            "**/BUILD",
        ],
    ),
    strip_prefix = "/docs",
    visibility = [
        "//scripts/docs:__pkg__",  # Scheuklappen: keep
    ],
)

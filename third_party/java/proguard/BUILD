licenses(["restricted"])  # GNU GPL v2

load("//src:release_archive.bzl", "release_archive")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//src:__pkg__",
        "//third_party:__pkg__",
    ],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "proguard6.2.2/docs/GPL.md",
        "proguard6.2.2/lib/proguard.jar",
    ],
    visibility = ["//src:__pkg__"],
)

release_archive(
    name = "proguard_zip",
    srcs = [
        "proguard6.2.2/docs/GPL.md",
        "proguard6.2.2/lib/proguard.jar",
    ],
    package_dir = "java_tools/third_party/java/proguard",
    visibility = ["//src:__pkg__"],
)

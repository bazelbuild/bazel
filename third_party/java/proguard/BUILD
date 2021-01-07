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
        "proguard5.3.3/docs/GPL.html",
        "proguard5.3.3/lib/proguard.jar",
    ],
    visibility = ["//src:__pkg__"],
)

release_archive(
    name = "proguard_zip",
    srcs = [
        "proguard5.3.3/docs/GPL.html",
        "proguard5.3.3/lib/proguard.jar",
    ],
    package_dir = "java_tools/third_party/java/proguard",
    visibility = ["//src:__pkg__"],
)

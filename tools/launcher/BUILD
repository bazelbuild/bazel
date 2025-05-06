load("//src:release_archive.bzl", "release_archive")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

alias(
    name = "launcher",
    actual = "//src/tools/launcher:launcher",
)

alias(
    name = "launcher_maker",
    actual = "//src/tools/launcher:launcher_maker",
)

release_archive(
    name = "launcher_prebuilt_all",
    srcs = [":launcher_maker"],
    package_dir = "test",
    visibility = ["//tools:__pkg__"],
)

release_archive(
    name = "launcher_prebuilt_windows",
    srcs = [":launcher"],
    package_dir = "test",
    visibility = ["//tools:__pkg__"],
    target_compatible_with = ["@bazel_tools//src/conditions:windows"],
)

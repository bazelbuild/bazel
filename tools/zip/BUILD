load("//src:release_archive.bzl", "release_archive")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

alias(
    name = "zipper",
    actual = "//third_party/ijar:zipper",
)

alias(
    name = "unzip_fdo",
    actual = "//third_party/ijar:zipper",
)

release_archive(
    name = "zip_prebuilt_all",
    srcs = [":zipper"],
    package_dir = "zip",
    visibility = ["//tools:__pkg__"],
)

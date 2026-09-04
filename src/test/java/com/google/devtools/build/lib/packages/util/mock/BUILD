package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib/packages:__pkg__"],
)

exports_files(
    [
        "android_cc_toolchain_config.bzl",
        "osx_cc_toolchain_config.bzl",
    ],
    visibility = ["//visibility:public"],
)

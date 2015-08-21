# Description:
#   The Bazel launcher.

filegroup(
    name = "blaze_util_os",
    srcs = select({
        "//src:darwin": [
            "blaze_util_darwin.cc",
            "blaze_util_posix.cc",
        ],
        "//conditions:default": [
            "blaze_util_linux.cc",
            "blaze_util_posix.cc",
        ],
    }),
)

cc_binary(
    name = "client",
    srcs = [
        "blaze.cc",
        "blaze_startup_options.cc",
        "blaze_startup_options.h",
        "blaze_startup_options_common.cc",
        "blaze_util.cc",
        "blaze_util.h",
        "blaze_util_platform.h",
        "option_processor.cc",
        "option_processor.h",
        ":blaze_util_os",
    ],
    copts = [
        "-Wno-sign-compare",
        "-DBLAZE_JAVA_CPU=\\\"k8\\\"",
    ],
    linkopts = select({
        "//src:darwin": [
        ],
        "//conditions:default": [
            "-lrt",
        ],
    }),
    visibility = ["//src:__pkg__"],
    deps = [
        "//src/main/cpp/util",
        "//src/main/cpp/util:md5",
        "//src/main/cpp/util:strings",
        "//third_party/ijar:zip",
    ],
)

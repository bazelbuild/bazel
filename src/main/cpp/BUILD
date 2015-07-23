# Description:
#   The Bazel launcher.

filegroup(
    name = "blaze_util_os",
    srcs = select({
        "//src:darwin": ["blaze_util_darwin.cc"],
        "//conditions:default": ["blaze_util_linux.cc"],
    }),
)

cc_binary(
    name = "client",
    srcs = [
        "blaze.cc",
        "blaze_startup_options.cc",
        "blaze_startup_options_common.cc",
        "blaze_util.cc",
        "option_processor.cc",
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

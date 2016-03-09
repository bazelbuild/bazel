# Description:
#   The Bazel launcher.
package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "blaze_util",
    srcs = [
        "blaze_util.cc",
    ] + select({
        "//src:darwin": [
            "blaze_util_darwin.cc",
            "blaze_util_posix.cc",
        ],
        "//src:darwin_x86_64": [
            "blaze_util_darwin.cc",
            "blaze_util_posix.cc",
        ],
        "//src:freebsd": [
            "blaze_util_freebsd.cc",
            "blaze_util_posix.cc",
        ],
        "//src:windows": [
            "blaze_util_mingw.cc",
        ],
        "//conditions:default": [
            "blaze_util_linux.cc",
            "blaze_util_posix.cc",
        ],
    }),
    hdrs = [
        "blaze_util.h",
        "blaze_util_platform.h",
    ],
    linkopts = select({
        "//src:darwin": [
        ],
        "//src:darwin_x86_64": [
        ],
        "//src:freebsd": [
        ],
        "//conditions:default": [
            "-lrt",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//src/main/cpp/util",
        "//src/main/cpp/util:blaze_exit_code",
    ],
)

cc_library(
    name = "blaze_abrupt_exit",
    srcs = [
        "blaze_abrupt_exit.cc",
    ],
    hdrs = [
        "blaze_abrupt_exit.h",
    ],
    deps = [
        "//src/main/cpp/util:blaze_exit_code",
    ],
)

cc_binary(
    name = "client",
    srcs = [
        "blaze.cc",
        "blaze_globals.h",
        "blaze_startup_options.cc",
        "blaze_startup_options.h",
        "blaze_startup_options_common.cc",
        "blaze_util_platform.h",
        "option_processor.cc",
        "option_processor.h",
    ],
    copts = [
        "-Wno-sign-compare",
        "-DBLAZE_JAVA_CPU=\\\"k8\\\"",
    ],
    linkopts = select({
        "//src:darwin": [
        ],
        "//src:darwin_x86_64": [
        ],
        "//src:freebsd": [
            "-lprocstat",
        ],
        "//conditions:default": [
            "-lrt",
        ],
    }),
    visibility = ["//src:__pkg__"],
    deps = [
        ":blaze_abrupt_exit",
        ":blaze_util",
        "//src/main/cpp/util",
        "//src/main/cpp/util:md5",
        "//src/main/cpp/util:strings",
        "//third_party/ijar:zip",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//src/main/cpp/util:srcs"],
    visibility = ["//src:__pkg__"],
)

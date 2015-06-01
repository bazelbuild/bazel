cc_library(
    name = "util",
    srcs = [
        "util/errors.cc",
        "util/file.cc",
        "util/numbers.cc",
        "util/port.cc",
        "util/strings.cc",
    ],
    hdrs = [
        "util/errors.h",
        "util/exit_code.h",
        "util/file.h",
        "util/numbers.h",
        "util/port.h",
        "util/strings.h",
    ],
    copts = [
        "-DBLAZE_OPENSOURCE=1",
    ],
    includes = ["."],
)

cc_library(
    name = "md5",
    srcs = ["util/md5.cc"],
    hdrs = ["util/md5.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
)

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
        "-DBLAZE_OPENSOURCE=1",
    ],
    includes = ["."],
    linkopts = select({
        "//src:darwin": [
        ],
        "//conditions:default": [
            "-lrt",
        ],
    }),
    visibility = ["//src:__pkg__"],
    deps = [
        ":md5",
        ":util",
        "//third_party/ijar:zip",
    ],
)

# Description:
#   C++ utility source for Bazel

cc_library(
    name = "util",
    srcs = [
        "errors.cc",
        "file.cc",
        "numbers.cc",
        "port.cc",
    ],
    hdrs = [
        "errors.h",
        "file.h",
        "numbers.h",
        "port.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":strings",
    ],
)

cc_library(
    name = "md5",
    srcs = ["md5.cc"],
    hdrs = ["md5.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "strings",
    srcs = ["strings.cc"],
    hdrs = ["strings.h"],
    # Automatically propagate the symbol definition to rules depending on this.
    defines = [
        "BLAZE_OPENSOURCE",
    ],
    visibility = ["//visibility:public"],
    deps = [":blaze_exit_code"],
)

cc_library(
    name = "blaze_exit_code",
    hdrs = ["exit_code.h"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/cpp:__pkg__"],
)

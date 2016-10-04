# Description:
#   C++ utility source for Bazel

cc_library(
    name = "util",
    hdrs = [
        "errors.h",
        "file.h",
        "numbers.h",
        "port.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":blaze_exit_code",
        ":errors",
        ":file",
        ":numbers",
        ":port",
        ":strings",
    ],
)

cc_library(
    name = "file_posix",
    srcs = ["file_posix.cc"],
    hdrs = ["file_posix.h"],
    visibility = ["//src/main/cpp:__pkg__"],
    deps = [
        ":blaze_exit_code",
        ":errors",
        ":file",
        ":strings",
    ],
)

cc_library(
    name = "file",
    srcs = ["file.cc"],
    hdrs = ["file.h"],
    visibility = ["//src/tools/singlejar:__pkg__"],
    deps = [
        ":blaze_exit_code",
        ":errors",
        ":strings",
    ],
)

cc_library(
    name = "errors",
    srcs = ["errors.cc"],
    hdrs = ["errors.h"],
    deps = [":port"],
)

cc_library(
    name = "port",
    srcs = ["port.cc"],
    hdrs = ["port.h"],
)

cc_library(
    name = "numbers",
    srcs = ["numbers.cc"],
    hdrs = ["numbers.h"],
    deps = [":strings"],
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

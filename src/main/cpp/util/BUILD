# Description:
#   C++ utility source for Bazel
package_group(
    name = "ijar",
    packages = [
        "//third_party/ijar/...",
    ],
)

cc_library(
    name = "util",
    hdrs = [
        "errors.h",
        "file.h",
        "file_platform.h",
        "md5.h",
        "numbers.h",
        "port.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":blaze_exit_code",
        ":errors",
        ":file",
        ":md5",
        ":numbers",
        ":port",
        ":strings",
    ],
)

cc_library(
    name = "file",
    srcs = ["file.cc"] + select({
        "//src:windows": [
            "file_windows.cc",
        ],
        "//src:windows_msys": [
            "file_windows.cc",
        ],
        "//src:windows_msvc": [
            "file_windows.cc",
        ],
        "//conditions:default": [
            "file_posix.cc",
        ],
    }),
    hdrs = [
        "file.h",
        "file_platform.h",
    ],
    visibility = [
        ":ijar",
        "//src/test/cpp/util:__pkg__",
        "//src/tools/singlejar:__pkg__",
    ],
    deps = [
        ":blaze_exit_code",
        ":errors",
        ":strings",
    ] + select({
        "//src:windows": ["//src/main/native:windows_jni_lib"],
        "//src:windows_msys": ["//src/main/native:windows_jni_lib"],
        "//src:windows_msvc": ["//src/main/native:windows_jni_lib"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "errors",
    srcs = ["errors.cc"] + select({
        "//src:windows": ["errors_windows.cc"],
        "//src:windows_msys": ["errors_windows.cc"],
        "//src:windows_msvc": ["errors_windows.cc"],
        "//conditions:default": ["errors_posix.cc"],
    }),
    hdrs = ["errors.h"],
    visibility = [
        ":ijar",
        "//src/main/cpp:__subpackages__",
    ],
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
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "bazel_log_handler",
    srcs = ["bazel_log_handler.cc"],
    hdrs = ["bazel_log_handler.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":file",
        ":logging",
    ],
)

cc_library(
    name = "md5",
    srcs = ["md5.cc"],
    hdrs = ["md5.h"],
    visibility = [
        "//src/main/native:__pkg__",
        "//src/test/cpp/util:__pkg__",
    ],
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

filegroup(
    name = "embedded_tools",
    srcs = glob(["*.cc"]) + glob(["*.h"]) + ["BUILD"],
    visibility = ["//visibility:public"],
)

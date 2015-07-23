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
    hdrs = [
        # This really belongs into its own library, but that doesn't work on the
        # Mac, because of issue #61 (header-only libraries don't build).
        "exit_code.h",
        "strings.h",
    ],
    # Automatically propagate the symbol definition to rules depending on this.
    defines = [
        "BLAZE_OPENSOURCE",
    ],
    visibility = ["//visibility:public"],
)

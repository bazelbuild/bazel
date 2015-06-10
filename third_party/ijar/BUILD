package(
    default_visibility = [
        "//src:__subpackages__",
        "//third_party/ijar:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "zip",
    srcs = ["zip.cc"],
    hdrs = [
        "common.h",
        "zip.h",
    ],
    # TODO(bazel-team): we should replace the -lz flag, it is non-hermetic.
    #  We should instead use a new_local_repository once the autoconf
    #  mechanism is ready.
    linkopts = ["-lz"],
)

cc_binary(
    name = "zipper",
    srcs = ["zip_main.cc"],
    visibility = ["//visibility:public"],
    deps = [":zip"],
)

cc_binary(
    name = "ijar",
    srcs = [
        "classfile.cc",
        "ijar.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [":zip"],
)

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
    # TODO(bazel-team): we should replace the -lz flag.
    linkopts = ["-lz"],
)

cc_binary(
    name = "zipper",
    srcs = ["zip_main.cc"],
    deps = [":zip"],
)

cc_binary(
    name = "ijar",
    srcs = [
        "classfile.cc",
        "ijar.cc",
    ],
    deps = [":zip"],
)

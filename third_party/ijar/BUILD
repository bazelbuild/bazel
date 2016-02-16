package(
    default_visibility = [
        "//src:__subpackages__",
        "//third_party/ijar:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//third_party:__pkg__"],
)

cc_library(
    name = "zip",
    srcs = [
        "mapped_file_unix.cc",
        "zip.cc",
    ],
    hdrs = [
        "common.h",
        "mapped_file.h",
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

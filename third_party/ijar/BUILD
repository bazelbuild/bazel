package(
    default_visibility = [
        "//src:__pkg__",
        "//third_party/ijar:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

cc_binary(
    name = "ijar",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    # TODO(bazel-team): we should replace the -lz flag.
    linkopts = ["-lz"],
)

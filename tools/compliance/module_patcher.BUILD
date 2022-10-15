# Note that we only evaluate this BUILD file in the context of the top level
# Bazel build, so it is not relative to the WORKSPACE file in this directory.

exports_files(glob(["*"]))

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        # "//tools:__subpackages__",
        "//visibility:public",
    ],
)

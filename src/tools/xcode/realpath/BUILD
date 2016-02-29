package(default_visibility = ["//src:__subpackages__"])

# This target will only build on a Mac.
genrule(
    name = "realpath_genrule",
    srcs = ["realpath.c"],
    outs = ["realpath"],
    cmd = "/usr/bin/xcrun clang -o $@ $<",
    output_to_bindir = 1,
    visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

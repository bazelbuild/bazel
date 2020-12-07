# This directory serves as the root of the builtins_bzl tree for Bazel, both in
# source form and as the value of --experimental_builtins_bzl_path.
#
# Because we use globs to gather builtins sources, there should be no
# subpackages here.

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

# A zipfile containing the builtins_bzl/ directory as it should appear in the
# package zip / install base.
genrule(
    name = "builtins_bzl_zip",
    srcs = [f for f in glob(["**"]) if f.endswith(".bzl")] + ["BUILD.builtins"],
    outs = ["builtins_bzl.zip"],
    # builtins_zip.sh zip output builtins_root files...
    cmd = "$(location //src:zip_builtins)" +
          " ''" +  # system zip
          " $@ src/main/starlark/builtins_bzl $(SRCS)",
    output_to_bindir = 1,
    tools = ["//src:zip_builtins"],
    visibility = ["//src:__pkg__"],
)

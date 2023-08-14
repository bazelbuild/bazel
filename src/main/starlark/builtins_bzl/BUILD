# This directory serves as the root of the builtins_bzl tree for Bazel, both in
# source form and at runtime when --experimental_builtins_bzl_path is set to
# %workspace%.
#
# Because we use globs to gather builtins sources, there should be no
# subpackages here.

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

# A zipfile containing the builtins_bzl/ directory, to be bundled as a Java
# resource with BazelRuleClassProvider.
genrule(
    name = "builtins_bzl_zip",
    srcs = glob(["**/*.bzl"]),
    outs = ["builtins_bzl.zip"],
    # builtins_zip.sh zip output builtins_root files...
    cmd = "$(location //src:zip_builtins)" +
          " ''" +  # system zip
          " $@ src/main/starlark/builtins_bzl $(SRCS)",
    message = "Building builtins_bzl.zip",
    output_to_bindir = 1,
    tools = ["//src:zip_builtins"],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib/bazel/rules:__pkg__",
    ],
)

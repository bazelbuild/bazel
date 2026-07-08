# This directory serves as the root of the builtins_bzl tree for Bazel, both in
# source form and at runtime when --experimental_builtins_bzl_path is set to
# %workspace%.
#
# Because we use globs to gather builtins sources, there should be no
# subpackages here.

load(":builtins_zip.bzl", "builtins_zip")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

# A zipfile containing the builtins_bzl/ directory, to be bundled as a Java
# resource with BazelRuleClassProvider.
builtins_zip(
    name = "builtins_bzl_zip",
    srcs = glob(
        ["**/*.bzl"],
        exclude = ["builtins_zip.bzl"],
    ),
    out = "builtins_bzl.zip",
    visibility = [
        "//src/main/java/com/google/devtools/build/lib/bazel/rules:__pkg__",
    ],
    zipper = "//third_party/ijar:zipper",
)

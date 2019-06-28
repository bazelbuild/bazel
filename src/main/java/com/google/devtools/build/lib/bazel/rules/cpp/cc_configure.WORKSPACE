load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe", "fail_with_message")

maybe(
    fail_with_message,
    "rules_cc",
    message = "Incompatible change `--incompatible_load_cc_rules_from_bzl` has been flipped, " +
    "and your WORKSPACE file doesn't contain @rules_cc repository. " +
    "See https://github.com/bazelbuild/bazel/issues/8743 for details.",
)

load("@bazel_tools//tools/cpp:cc_configure.bzl", "cc_configure")

cc_configure()

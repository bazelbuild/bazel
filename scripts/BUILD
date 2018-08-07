# Scripts for IDE / Environment support for Bazel
package(default_visibility = ["//visibility:private"])

# The master copy of the completion logic are bazel-complete-template.bash
# and bazel-complete-header.bash.
# This is where you should make edits.
genrule(
    name = "bash_completion",
    srcs = ["bazel-complete-template.bash"],
    outs = ["bazel-complete.bash"],
    cmd = " ".join([
        "$(location :generate_bash_completion.sh)",
        "--bazel=$(location //src:bazel)",
        "--javabase=$(JAVABASE)",
        "--output=$@",
        "--prepend=$(location bazel-complete-template.bash)",
    ]),
    output_to_bindir = 1,
    # Bazel 0.8.0 doesn't have this target under @bazel_tools, so we have to
    # use it from the main repository
    toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
    tools = [
        ":generate_bash_completion.sh",
        "//src:bazel",
        "//tools/defaults:jdk",
    ],
    visibility = ["//scripts/packages:__subpackages__"],
)

sh_test(
    name = "bash_completion_test",
    size = "small",
    srcs = ["bash_completion_test.sh"],
    data = [
        "bazel-complete.bash",
        "testenv.sh",
        "//src/test/shell:bashunit",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//scripts/docs:srcs",
        "//scripts/packages:srcs",
        "//scripts/release:srcs",
    ],
    visibility = ["//:__pkg__"],
)

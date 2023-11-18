# Scripts for IDE / Environment support for Bazel
package(default_visibility = ["//visibility:private"])

# The master copy of the completion logic lives in bazel-complete-header.bash
# and bazel-complete-template.bash. You should make edits in those files.
genrule(
    name = "bash_completion",
    srcs = [
        "bazel-complete-header.bash",
        "bazel-complete-template.bash",
    ],
    outs = ["bazel-complete.bash"],
    cmd = " ".join([
        "$(location :generate_bash_completion.sh)",
        "--bazel=$(location //src:bazel)",
        "--output=$@",
        "--prepend=$(location bazel-complete-header.bash)",
        "--prepend=$(location bazel-complete-template.bash)",
    ]),
    output_to_bindir = 1,
    tools = [
        ":generate_bash_completion.sh",
        "//src:bazel",
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
        "@bazel_tools//tools/bash/runfiles",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//scripts/docs:srcs",
        "//scripts/packages:srcs",
        "//scripts/release:srcs",
        "//scripts/zsh_completion:srcs",
    ],
    visibility = ["//:__pkg__"],
)

py_binary(
    name = "generate_fish_completion",
    srcs = ["generate_fish_completion.py"],
    deps = ["//third_party/py/abseil"],
)

genrule(
    name = "fish_completion",
    outs = ["bazel.fish"],
    cmd = " ".join([
        "$(location :generate_fish_completion)",
        "--bazel=$(location //src:bazel)",
        "--output=$@",
    ]),
    tools = [
        ":generate_fish_completion",
        "//src:bazel",
    ],
    visibility = ["//scripts/packages:__subpackages__"],
)

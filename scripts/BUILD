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
    visibility = [
        "//scripts/packages:__subpackages__",
        "//src/test/py/bazel:__pkg__",
    ],
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
    name = "jq_linux",
    srcs = select({
        "@platforms//cpu:arm64": ["@jq_linux_arm64//file"],
        "@platforms//cpu:x86_64": ["@jq_linux_amd64//file"],
        "@platforms//cpu:s390x": ["@jq_linux_s390x//file"],
    }),
)

filegroup(
    name = "jq_macos",
    srcs = select({
        "@platforms//cpu:arm64": ["@jq_macos_arm64//file"],
        "@platforms//cpu:x86_64": ["@jq_macos_amd64//file"],
    }),
)

filegroup(
    name = "jq",
    srcs = select({
        "@platforms//os:linux": [":jq_linux"],
        "@platforms//os:macos": [":jq_macos"],
        "@platforms//os:windows": ["@jq_windows_amd64//file"],
    }),
)

sh_test(
    name = "bazel_lockfile_merge_test",
    size = "small",
    srcs = ["bazel_lockfile_merge_test.sh"],
    data = [
        "bazel-lockfile-merge.jq",
        "testenv.sh",
        ":jq",
        "//src/test/shell:bashunit",
        "//src/test/tools/bzlmod:MODULE.bazel.lock",
        "@bazel_tools//tools/bash/runfiles",
    ],
    env = {
        "JQ_RLOCATIONPATH": "$(rlocationpath :jq)",
    },
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

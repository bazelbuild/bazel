# Allowlists for restricting access to Bazel features.

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "//tools/allowlists/android_binary_allowlist:srcs",
        "//tools/allowlists/config_feature_flag:srcs",
        "//tools/allowlists/extend_rule_allowlist:srcs",
        "//tools/allowlists/function_transition_allowlist:srcs",
        "//tools/whitelists:srcs",
    ],
    visibility = ["//tools:__pkg__"],
)

# Allowlists for restricting access to Bazel features.

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "//tools/allowlists/config_feature_flag:srcs",
        "//tools/allowlists/function_transition_allowlist:srcs",
        "//tools/whitelists:srcs",
    ],
    visibility = ["//tools:__pkg__"],
)

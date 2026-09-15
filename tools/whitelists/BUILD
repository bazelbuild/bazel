# Allowlists for restricting access to Bazel features.

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "//tools/whitelists/config_feature_flag:srcs",
        "//tools/whitelists/function_transition_whitelist:srcs",
    ],
    visibility = ["//tools/allowlists:__pkg__"],
)

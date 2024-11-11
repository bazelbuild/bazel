# Allowlists for restricting access to Bazel features.

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "//tools/allowlists/config_feature_flag:srcs",
        "//tools/allowlists/dormant_dependency_allowlist:srcs",
        "//tools/allowlists/extend_rule_allowlist:srcs",
        "//tools/allowlists/function_transition_allowlist:srcs",
        "//tools/allowlists/initializer_allowlist:srcs",
        "//tools/allowlists/skip_validations_allowlist:srcs",
        "//tools/allowlists/subrules_allowlist:srcs",
        "//tools/whitelists:srcs",
    ],
    visibility = ["//tools:__pkg__"],
)

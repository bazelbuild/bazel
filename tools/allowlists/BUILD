# Allowlists for restricting access to Bazel features.

ALLOWLISTS = [
    "//tools/allowlists/config_feature_flag:srcs",
    "//tools/allowlists/dormant_dependency_allowlist:srcs",
    "//tools/allowlists/extend_rule_allowlist:srcs",
    "//tools/allowlists/function_transition_allowlist:srcs",
    "//tools/allowlists/initializer_allowlist:srcs",
    "//tools/allowlists/materializer_rule_allowlist:srcs",
    "//tools/allowlists/subdirectory_allowlist:srcs",
    "//tools/allowlists/subrules_allowlist:srcs",
]

filegroup(
    name = "srcs",
    srcs = ["BUILD"] + ALLOWLISTS,
    visibility = ["//tools:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = ALLOWLISTS,
    visibility = ["//tools:__pkg__"],
)

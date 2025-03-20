# Description:
#   Package groups for restricting access to config_feature_flag to specific
#   packages, allowing for careful rollout as it is an experimental feature.

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/allowlists:__pkg__"],
)

package_group(
    name = "config_feature_flag",
    packages = ["public"],
)

package_group(
    name = "config_feature_flag_setter",
    packages = ["public"],
)

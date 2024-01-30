# Description:
#   Package groups for restricting access to android_binary_allowlist to specific
#   packages, allowing for careful rollout as it is an experimental feature.

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/allowlists:__pkg__"],
)

package_group(
    name = "enable_starlark_dex_desugar_proguard",
    packages = [],
)
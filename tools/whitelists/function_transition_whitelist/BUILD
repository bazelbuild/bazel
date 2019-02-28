# Description:
#   Package group restricting access to starlark-defined transitions, allowing for careful rollout as it is an experimental feature.

package_group(
    name = "function_transition_whitelist",
    packages = ["//..."],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/whitelists:__pkg__"],
)

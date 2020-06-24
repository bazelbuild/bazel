# Description:
#   Package group restricting access to starlark-defined transitions, allowing for careful rollout as it is an experimental feature.

package_group(
    name = "function_transition_allowlist",
    packages = ["//..."],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/allowlists:__pkg__"],
)

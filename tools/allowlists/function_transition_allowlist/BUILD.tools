# Description:
#   Package group restricting access to starlark-defined transitions.

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/allowlists:__pkg__"],
)

package_group(
    name = "function_transition_allowlist",
    packages = ["public"],
)

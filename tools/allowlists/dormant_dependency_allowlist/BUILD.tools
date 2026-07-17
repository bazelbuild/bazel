# Package group restricting access to dormant dependencies.

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/allowlists:__pkg__"],
)

package_group(
    name = "dormant_dependency_allowlist",
    packages = ["public"],
)
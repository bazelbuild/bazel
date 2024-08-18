# Description:
#   Package group restricting access to extended rules, allowing
#   for careful rollout as it is an experimental feature.

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/allowlists:__pkg__"],
)

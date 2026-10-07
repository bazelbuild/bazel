# Description:
#   Package group restricting access to declare_subdirectory in map_directory,
#   allowing for careful rollout as it is an experimental feature.

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/allowlists:__pkg__"],
)

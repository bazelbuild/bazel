load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

maybe(
    local_repository,
    name = "platforms",
    path = __embedded_dir__ + "/platforms",
)

local_config_platform(name = "local_config_platform")

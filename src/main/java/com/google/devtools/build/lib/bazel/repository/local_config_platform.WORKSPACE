load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

maybe(
    local_repository,
    "platforms",
    path = __embedded_dir__ + "/platforms",
)

maybe(
    local_repository,
    "internal_platforms_do_not_use",
    path = __embedded_dir__ + "/platforms",
)

maybe(
    local_config_platform,
    "local_config_platform",
)

load("@internal_platforms_do_not_use//host:extension.bzl", "host_platform_repo")
maybe(
    host_platform_repo,
    "host_platform",
)

# Prebuilt tools that @bazel_tools downloads rather than embeds, so that they
# are available for all execution and target platforms regardless of the host.
# See prebuilt_tools_extension in //tools/test:extensions.bzl.

load("//src:release_archive.bzl", "release_archive")

package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

# Keep in sync with PREBUILT_TOOLS_PLATFORMS in //tools:build_defs.bzl.
_PLATFORMS = {
    "//src/conditions:darwin_arm64": "darwin_arm64",
    "//src/conditions:darwin_x86_64": "darwin_x86_64",
    "//src/conditions:linux_aarch64": "linux_aarch64",
    "//src/conditions:linux_x86_64": "linux_x86_64",
    "//src/conditions:windows_arm64": "windows_arm64",
    "//src/conditions:windows_x86_64": "windows_x86_64",
}

# The platform that :prebuilt_tools_zip is built for.
genrule(
    name = "platform",
    outs = ["platform.txt"],
    cmd = select({
        condition: "echo {} > $@".format(platform)
        for condition, platform in _PLATFORMS.items()
    }),
    visibility = ["//src/test:__subpackages__"],
)

release_archive(
    name = "windows_tools_zip",
    src_map = {
        "//src/tools/launcher": "launcher.exe",
        "//src/tools/launcher:launcher_maker": "launcher_maker.exe",
        "//tools/test:tw": "tw.exe",
        "//tools/test:xml": "xml.exe",
    },
)

release_archive(
    name = "unix_tools_zip",
    src_map = {
        "//src/tools/launcher:launcher_maker": "launcher_maker",
    },
)

release_archive(
    name = "prebuilt_tools_zip",
    src_map = {
        "BUILD.prebuilt_tools": "BUILD",
    },
    visibility = ["//src/test:__subpackages__"],
    deps = select({
        "//src/conditions:windows": [":windows_tools_zip"],
        "//conditions:default": [":unix_tools_zip"],
    }),
)

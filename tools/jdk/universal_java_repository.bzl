_PLATFORMS = {
    "linux_x86_64": [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    "macos_aarch64": [
        "@platforms//os:macos",
        "@platforms//cpu:aarch64",
    ],
    "universal": [
        "@platforms//:universal",
    ],
}

_TOOLS = ["java", "javac"]
_MERGED_DIRS = ["include"]
_DUMMIES = ["lib/jrt-fs.jar", "modules"]

def _get_build_file(ctx):
    return """
load("@rules_java//java:defs.bzl", "java_runtime")

java_runtime(
    name = "jdk",
    srcs = glob(["bin/**", "include/**", "lib/**", "modules"], allow_empty = True) + [
        "@{linux_repo}//:jdk",
        "@{macos_repo}//:jdk",
    ],
)
""".format(
        linux_repo = ctx.attr.linux_repo,
        macos_repo = ctx.attr.macos_repo,
    )

def _get_tool_wrapper(ctx, tool):
    return """#!/bin/bash
external=$(dirname -- "${{BASH_SOURCE[0]}}")/../..
if [[ "$OSTYPE" == linux* ]]; then
    exec "$external/{linux_repo}/bin/{tool}" "$@"
elif [[ "$OSTYPE" == darwin* ]]; then
    exec "$external/{macos_repo}/bin/{tool}" "$@"
else
    echo "Unknown platform $OSTYPE" >&2
    exit 1
fi
""".format(linux_repo = ctx.attr.linux_repo, macos_repo = ctx.attr.macos_repo, tool = tool)

def _universal_java_repo_impl(ctx):
    ctx.file("WORKSPACE", "workspace(name = \"{name}\")\n".format(name = ctx.name))
    ctx.file("BUILD", _get_build_file(ctx))

    for tool in _TOOLS:
        ctx.file("bin/{tool}".format(tool = tool), _get_tool_wrapper(ctx, tool), executable = True)
    for dir in _MERGED_DIRS:
        ctx.execute(["/bin/bash", "-c", """
            set -e
            mkdir -p include
            ln -s ../../{linux_repo}/include/* ../../{macos_repo}/include/* include
            """.format(linux_repo = ctx.attr.linux_repo, macos_repo = ctx.attr.macos_repo)])
    for dummy in _DUMMIES:
        ctx.file(dummy)

_universal_java_repo = repository_rule(
    local = True,
    implementation = _universal_java_repo_impl,
    attrs = {
        "linux_repo": attr.string(),
        "macos_repo": attr.string(),
    },
)

def _toolchain_config_impl(ctx):
    ctx.file("WORKSPACE", "workspace(name = \"{name}\")\n".format(name = ctx.name))
    ctx.file("BUILD.bazel", ctx.attr.build_file)

_toolchain_config = repository_rule(
    local = True,
    implementation = _toolchain_config_impl,
    attrs = {
        "build_file": attr.string(),
    },
)

def universal_java_repository(name, version, prefix = "remotejdk"):
    _universal_java_repo(
        name = name,
        linux_repo = "{prefix}{version}_linux".format(
            prefix = prefix,
            version = version,
        ),
        macos_repo = "{prefix}{version}_macos_aarch64".format(
            prefix = prefix,
            version = version,
        ),
    )

    config_repo_name = "{name}_toolchain_config_repo".format(name = name)

    config_repo_build_file = """
config_setting(
    name = "prefix_version_setting",
    values = {{"java_runtime_version": "{prefix}_{version}"}},
    visibility = ["//visibility:private"],
)
config_setting(
    name = "version_setting",
    values = {{"java_runtime_version": "{version}"}},
    visibility = ["//visibility:private"],
)
alias(
    name = "version_or_prefix_version_setting",
    actual = select({{
        ":version_setting": ":version_setting",
        "//conditions:default": ":prefix_version_setting",
    }}),
    visibility = ["//visibility:private"],
)
""".format(prefix = prefix, version = version)

    for platform_name, target_compatible_with in _PLATFORMS.items():
        registration_name = "toolchain_{}".format(platform_name)

        config_repo_build_file += """
toolchain(
    name = "{registration_name}",
    exec_compatible_with = ["@platforms//:universal"],
    target_compatible_with = {target_compatible_with},
    target_settings = [":version_or_prefix_version_setting"],
    toolchain_type = "@bazel_tools//tools/jdk:runtime_toolchain_type",
    toolchain = "{toolchain}",
)""".format(
            toolchain = "@{name}//:jdk".format(name = name),
            registration_name = registration_name,
            target_compatible_with = target_compatible_with,
        )

        native.register_toolchains("@{config_repo_name}//:{registration_name}".format(
            config_repo_name = config_repo_name,
            registration_name = registration_name,
        ))

    _toolchain_config(
        name = config_repo_name,
        build_file = config_repo_build_file,
    )

def _universal_split_transition_impl(ctx, attr):
    return {
        "x86_64" : {
            "//command_line_option:cpu": "darwin_x86_64",
        },
        "arm64" : {
            "//command_line_option:cpu": "darwin_arm64",
        }
    }

_universal_split_transition = transition(
    implementation = _universal_split_transition_impl,
    inputs = [],
    outputs = ["//command_line_option:cpu"],
)

def _impl(ctx):
    binaries = [
        attr.files.to_list()[0] for attr in ctx.split_attr.binary.values()
    ]
    out = ctx.actions.declare_file(ctx.label.name + "/" + ctx.attr.output_name)
    args = ctx.actions.args()
    args.add("-create")
    args.add_all(binaries)
    args.add("-output", out)
    apple_env = {}
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]
    apple_env.update(apple_common.apple_host_system_env(xcode_config))
    apple_env.update(
        apple_common.target_apple_env(
            xcode_config,
            ctx.fragments.apple.multi_arch_platform(apple_common.platform_type.macos),
        ),
    )
    ctx.actions.run(
        executable = "/usr/bin/lipo",
        arguments = [args],
        inputs = binaries,
        outputs = [out],
        execution_requirements = xcode_config.execution_info(),
        env = apple_env,
    )
    return DefaultInfo(executable = out)

darwin_universal_binary = rule(
    implementation = _impl,
    attrs = {
        "output_name" : attr.string(),
        "binary": attr.label(cfg = _universal_split_transition),
        "_xcode_config": attr.label(default = "@bazel_tools//tools/osx:current_xcode_config"),
        "_allowlist_function_transition": attr.label(default = "@bazel_tools//tools/allowlists/function_transition_allowlist"),
    },
    fragments = ["apple"],
)

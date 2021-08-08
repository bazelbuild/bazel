"""A toolchain for Protobuf rules."""

load("//tools/config:common_settings.bzl", "BuildSettingInfo")

def _proto_toolchain_impl(ctx):
    compiler = ctx.attr.compiler
    compiler_options = ctx.attr.compiler_options

    # TODO(yannic): Migrate native `--protocopt` to Starlark flag when
    # accumulating multiple flags to a list works correctly.
    command_line_compiler_options = getattr(ctx.fragments.proto, "protoc_opts", [])

    return [
        platform_common.ToolchainInfo(
            proto = ProtoToolchainInfo(
                compiler = compiler.files_to_run,
                compiler_options = compiler_options + command_line_compiler_options,
            ),
        ),
    ]

proto_toolchain = rule(
    implementation = _proto_toolchain_impl,
    attrs = {
        "compiler": attr.label(
            mandatory = True,
            doc = "The proto compiler to use.",
        ),
        "compiler_options": attr.string_list(
            doc = "Additional options to pass to `compiler`.",
        ),
    },
    provides = [
        platform_common.ToolchainInfo,
    ],
    doc = """
Represents a `Protobuf` toolchain.

This rule is responsible for:
  - Providing the `protoc` binary for compiling `.proto` files to descript sets
    and generating, e.g., `C++` or `Java` code.
  - Providing common command-line options to pass to all invocations of
    `compiler`.
""",
    fragments = [
        "proto",
    ],
)

"""An alias for `proto_toolchain`.

For migration only! This alias rule allow (native) rules to depend on
`@bazel_tools//tools/proto:toolchain_type` based on
`--incompatible_proto_use_toolchain_resolution` (i.e. without decalaring a
toolchain dependency on it, which cannot be done based on a configuration flag).
"""

def _proto_toolchain_alias_impl(ctx):
    toolchain_info = ctx.toolchains["@bazel_tools//tools/proto:toolchain_type"]
    return [
        toolchain_info.proto,
    ]

proto_toolchain_alias = rule(
    implementation = _proto_toolchain_alias_impl,
    provides = [
        ProtoToolchainInfo,
    ],
    toolchains = [
        "@bazel_tools//tools/proto:toolchain_type",
    ],
)

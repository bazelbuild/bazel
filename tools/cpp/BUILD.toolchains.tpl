load("@local_config_platform//:constraints.bzl", "HOST_CONSTRAINTS")
toolchain(
    name = "cc-toolchain-%{name}",
    exec_compatible_with = HOST_CONSTRAINTS,
    target_compatible_with = HOST_CONSTRAINTS,
    toolchain = "@local_config_cc//:cc-compiler-%{name}",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

toolchain(
    name = "cc-toolchain-armeabi-v7a",
    exec_compatible_with = HOST_CONSTRAINTS,
    target_compatible_with = [
        "@bazel_tools//platforms:arm",
        "@bazel_tools//platforms:android",
    ],
    toolchain = "@local_config_cc//:cc-compiler-armabi-v7a",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)


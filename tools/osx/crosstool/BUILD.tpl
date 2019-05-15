package(default_visibility = ["//visibility:public"])

load(":osx_archs.bzl", "OSX_TOOLS_ARCHS", "OSX_TOOLS_CONSTRAINTS")
load(":cc_toolchain_config.bzl", "cc_toolchain_config")

CC_TOOLCHAINS = [(
    cpu + "|compiler",
    ":cc-compiler-" + cpu,
) for cpu in OSX_TOOLS_ARCHS] + [(
    cpu,
    ":cc-compiler-" + cpu,
) for cpu in OSX_TOOLS_ARCHS] + [
    ("k8|compiler", ":cc-compiler-darwin_x86_64"),
    ("darwin|compiler", ":cc-compiler-darwin_x86_64"),
    ("k8", ":cc-compiler-darwin_x86_64"),
    ("darwin", ":cc-compiler-darwin_x86_64"),
]

cc_library(
    name = "malloc",
)

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
    name = "cc_wrapper",
    srcs = ["cc_wrapper.sh"],
)

cc_toolchain_suite(
    name = "toolchain",
    toolchains = dict(CC_TOOLCHAINS),
)

[
    filegroup(
        name = "osx_tools_" + arch,
        srcs = [
            ":cc_wrapper",
            ":libtool",
            ":make_hashed_objlist.py",
            ":wrapped_ar",
            ":wrapped_clang",
            ":wrapped_clang_pp",
            ":xcrunwrapper.sh",
        ],
    )
    for arch in OSX_TOOLS_ARCHS
]

[
    apple_cc_toolchain(
        name = "cc-compiler-" + arch,
        all_files = ":osx_tools_" + arch,
        ar_files = ":empty",
        as_files = ":empty",
        compiler_files = ":osx_tools_" + arch,
        dwp_files = ":empty",
        linker_files = ":osx_tools_" + arch,
        objcopy_files = ":empty",
        strip_files = ":osx_tools_" + arch,
        supports_param_files = 0,
        toolchain_config = ":" + (
            arch if arch != "armeabi-v7a" else "stub_armeabi-v7a"
        ),
        toolchain_identifier = (
            arch if arch != "armeabi-v7a" else "stub_armeabi-v7a"
        ),
    )
    for arch in OSX_TOOLS_ARCHS
]

[
    cc_toolchain_config(
        name = (arch if arch != "armeabi-v7a" else "stub_armeabi-v7a"),
        compiler = "compiler",
        cpu = arch,
    )
    for arch in OSX_TOOLS_ARCHS
]

[
    toolchain(
        name = "cc-toolchain-" + arch,
        exec_compatible_with = [
            # These only execute on macOS.
            "@bazel_tools//platforms:osx",
            "@bazel_tools//platforms:x86_64",
        ],
        target_compatible_with = OSX_TOOLS_CONSTRAINTS[arch],
        toolchain = ":cc-compiler-" + arch,
        toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
    )
    for arch in OSX_TOOLS_ARCHS
]

def _impl(ctx):
    ctx.file("BUILD", """
load("@bazel_tools//tools/universal_binary:universal_binary.bzl", "universal_binary")

package(default_visibility = ["//visibility:public"])

alias(
    name = "has_androidsdk",
    actual = "@bazel_tools//tools/android:always_true",
)

[universal_binary(
    name = "%s_binary" % tool,
    linux = "@{linux_repo}//:%s_binary" % tool,
    darwin = "@{darwin_repo}//:%s_binary" % tool,
) for tool in ["aapt", "aapt2", "aidl", "zipalign"]]

[universal_binary(
    name = "%s_binary" % tool,
    linux = "@{linux_repo}//:%s" % tool,
    darwin = "@{darwin_repo}//:%s" % tool,
) for tool in ["adb", "apksigner", "generate_main_dex_list", "dx"]]

android_sdk(
    name = "sdk-{api_level}",
    build_tools_version = "{build_tools_version}",
    proguard = "@bazel_tools//tools/jdk:proguard",
    aapt = ":aapt_binary",
    aapt2 = ":aapt2_binary",
    dx = ":dx_universal",
    legacy_main_dex_list_generator = ":generate_main_dex_list",
    adb = ":adb_binary",
    framework_aidl = "@{linux_repo}//:framework_aidl-{api_level}",
    aidl = ":aidl_binary",
    android_jar = "@{linux_repo}//:android_jar-{api_level}",
    shrinked_android_jar = "@{linux_repo}//:shrinked_android_jar-{api_level}",
    main_dex_classes = "@{linux_repo}//:main_dex_classes",
    apksigner = ":apksigner_binary",
    zipalign = ":zipalign_binary",
)

toolchain(
    name = "sdk-{api_level}-toolchain",
    toolchain_type = "@bazel_tools//tools/android:sdk_toolchain_type",
    exec_compatible_with = [
        "@platforms//:universal",
    ],
    target_compatible_with = [
        "@platforms//os:android",
    ],
    toolchain = ":sdk-{api_level}",
)

alias(
    name = "sdk",
    actual = ":sdk-{api_level}",
)
""".format(
        name = ctx.name,
        api_level = ctx.attr.api_level,
        build_tools_version = ctx.attr.build_tools_version,
        linux_repo = ctx.attr.linux_repo,
        darwin_repo = ctx.attr.darwin_repo,
    ))

_universal_android_sdk_repository = repository_rule(
    implementation = _impl,
    attrs = {
        "api_level": attr.int(mandatory = True),
        "build_tools_version": attr.string(mandatory = True),
        "linux_repo": attr.string(mandatory = True),
        "darwin_repo": attr.string(mandatory = True),
    },
)

def universal_android_sdk_repository(name, api_level, build_tools_version, linux_repo, darwin_repo):
    _universal_android_sdk_repository(
        name = name,
        api_level = api_level,
        build_tools_version = build_tools_version,
        linux_repo = linux_repo,
        darwin_repo = darwin_repo,
    )

    native.bind(
        name = "android/sdk",
        actual = "@%s//:sdk" % name,
    )

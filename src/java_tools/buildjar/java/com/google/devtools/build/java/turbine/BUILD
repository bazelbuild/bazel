load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@rules_graalvm//graalvm:defs.bzl", "native_image")
load("@rules_java//java:defs.bzl", "java_binary", "java_library")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
load(":platform_aliases.bzl", "linux_aarch64_glibc_alias")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src/java_tools/buildjar:buildjar_package_group"],
)

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

_TURBINE_MAIN_CLASS = "com.google.turbine.main.Main"

java_library(
    name = "turbine_deps",
    runtime_deps = [
        "//src/main/protobuf:deps_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:turbine",
    ],
)

java_binary(
    name = "turbine_direct_binary",
    main_class = _TURBINE_MAIN_CLASS,
    runtime_deps = [":turbine_deps"],
)

# If enabled, build Turbine with Oracle GraalVM rather than GraalVM Community.
# See https://www.graalvm.org/downloads/ for licensing restrictions.
bool_flag(
    name = "use_oracle_graalvm",
    build_setting_default = False,
)

config_setting(
    name = "do_use_oracle_graalvm",
    flag_values = {"use_oracle_graalvm": str(True)},
)

# If enabled, build Turbine with PGO instrumentation and without a PGO profile.
# This is used by :update_turbine_pgo. Requires the use of Oracle GraalVM.
bool_flag(
    name = "pgo_instrument",
    build_setting_default = False,
)

config_setting(
    name = "do_pgo_instrument",
    flag_values = {
        "pgo_instrument": str(True),
        "use_oracle_graalvm": str(True),
    },
)

config_setting(
    name = "linux_aarch64_musl",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
        "@llvm//constraints/libc:musl",
    ],
)

config_setting(
    name = "linux_musl",
    constraint_values = [
        "@platforms//os:linux",
        "@llvm//constraints/libc:musl",
    ],
)

alias(
    name = "turbine_direct_graal",
    actual = select({
        # GraalVM only ships the static JDK libraries for musl on x86_64.
        ":linux_aarch64_musl": ":turbine_direct_graal_linux_aarch64_glibc",
        "@platforms//os:windows": ":turbine_direct_graal_with_app_manifest",
        "//conditions:default": ":turbine_direct_graal_unpatched",
    }),
)

linux_aarch64_glibc_alias(
    name = "turbine_direct_graal_linux_aarch64_glibc",
    actual = ":turbine_direct_graal_unpatched",
    tags = ["manual"],
)

native_image(
    name = "turbine_direct_graal_unpatched",
    # Native Image on Windows only supports MSVC, whose flags conflict with the ones it adds.
    cc_toolchain_flags = select({
        "@platforms//os:windows": False,
        "//conditions:default": True,
    }),
    executable_name = select({
        # TODO(cushon): restore .exe suffix on windows
        # see https://github.com/sgammon/rules_graalvm/issues/324
        "@bazel_tools//src/conditions:windows": "turbine_direct_graal_unpatched",
        "//conditions:default": "turbine_direct_graal",
    }),
    extra_args = [
        # Opt into stricter behavior to squelch a warning on recent GraalVM versions.
        "--strict-image-heap",
        # Workaround for https://github.com/oracle/graal/issues/4757.
        "-H:-UseContainerSupport",
        # A benchmark on Bazel itself shows a ~15% improvement in combined compile and header
        # compile action time on an incremental build triggered by a signature change to Label with
        # this option. 256m provides a noticeably smaller improvement, higher values do not provide
        # further improvement and would go over the local resource estimate in
        # com.google.devtools.build.lib.rules.java.JavaCompileAction.LOCAL_RESOURCES.
        # See :turbine_benchmark for the benchmark script used.
        "-R:MinHeapSize=512m",
    ] + select({
        ":linux_musl": [
            "--static",
            "--libc=musl",
        ],
        "@platforms//os:linux": [
            # Statically link zlib but not glibc.
            "-H:+StaticExecutableWithDynamicLibC",
            # Native Image's version script hides _IO_stdin_used, which glibc references. Unlike
            # GNU ld, lld reports this.
            "-H:NativeLinkerOption=-Wl,--allow-shlib-undefined",
        ],
        "@platforms//os:windows": [
            # The charset specified by sun.jnu.encoding is not automatically included in the image,
            # but may be one of the legacy code pages on Windows, which aren't added by default.
            # https://github.com/oracle/graal/pull/10232
            "-H:+AddAllCharsets",
        ],
        "//conditions:default": [],
    }) + select({
        # Graal's default settings result in executables that aren't sufficiently compatible for
        # general use in Bazel.
        "@platforms//cpu:x86_64": [
            "-march=x86-64-v2",
        ],
        "@platforms//cpu:aarch64": [
            "-march=armv8-a",
        ],
        "//conditions:default": [],
    }) + select({
        ":do_pgo_instrument": ["--pgo-instrument"],
        "//conditions:default": [],
    }),
    main_class = _TURBINE_MAIN_CLASS,
    native_image_tool = select({
        ":do_use_oracle_graalvm": "@graalvm_oracle//:native-image",
        "//conditions:default": None,
    }),
    profiles = select({
        ":do_pgo_instrument": [],
        ":do_use_oracle_graalvm": ["profile.iprof"],
        "//conditions:default": [],
    }),
    # This provides libz.a on Linux instead of the host system.
    static_zlib = "@zlib",
    deps = [":turbine_deps"],
)

# On Windows, add an app manifest to the binary to force it to run with a UTF-8
# code page. It is built with one, but without the app manifest it will not be
# able to use UTF-8 for filesystem operations.
# https://github.com/oracle/graal/issues/10237
genrule(
    name = "turbine_direct_graal_with_app_manifest",
    srcs = [
        ":turbine_direct_graal_unpatched",
        "turbine_direct_graal.manifest",
    ],
    outs = ["turbine_direct_graal.exe"],
    cmd = """\
cp $(location :turbine_direct_graal_unpatched) $@
chmod +w $@
cat $(location turbine_direct_graal.manifest) | $(location //src:write_manifest) $@
""",
    target_compatible_with = ["@platforms//os:windows"],
    tools = ["//src:write_manifest"],
)

# Run with -c opt.
sh_binary(
    name = "turbine_benchmark",
    srcs = ["turbine_benchmark.sh"],
    args = ["$(rlocationpath turbine_direct_graal)"],
    data = [":turbine_direct_graal"],
    deps = ["@bazel_tools//tools/bash/runfiles"],
)

# Run this target to update profile.iprof.
sh_binary(
    name = "update_turbine_pgo",
    srcs = ["update_turbine_pgo.sh"],
)

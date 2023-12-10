load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@rules_graalvm//graalvm:defs.bzl", "native_image")
load("@rules_java//java:defs.bzl", "java_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src/java_tools/buildjar:buildjar_package_group"],
)

licenses(["notice"])

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

# If enabled, build Turbine with PGO instrumentation and without a PGO profile.
# This is used by :update_turbine_pgo.
bool_flag(
    name = "turbine_direct_graal_pgo_instrument",
    build_setting_default = False,
)

config_setting(
    name = "turbine_direct_graal_do_pgo_instrument",
    flag_values = {"turbine_direct_graal_pgo_instrument": str(True)},
)

native_image(
    name = "turbine_direct_graal",
    executable_name = select({
        # TODO(cushon): restore .exe suffix on windows
        # see https://github.com/sgammon/rules_graalvm/issues/324
        "@bazel_tools//src/conditions:windows": "%target%",
        "//conditions:default": "%target%",
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
        "@platforms//os:linux": [
            # Statically link zlib but not glibc.
            "-H:+StaticExecutableWithDynamicLibC",
        ],
        "@platforms//os:windows": [
            # The charset specified by sun.jnu.encoding is not automatically included in the image,
            # but may be one of the legacy code pages on Windows, which aren't added by default.
            # https://github.com/oracle/graal/pull/10232
            "-H:+AddAllCharsets",
        ],
        "//conditions:default": [],
    }) + select({
        "@platforms//cpu:x86_64": [
            # Graal's default settings result in executables that aren't sufficiently compatible for
            # general use in Bazel.
            "-march=x86-64-v2",
        ],
        "//conditions:default": [],
    }) + select({
        ":turbine_direct_graal_do_pgo_instrument": ["--pgo-instrument"],
        "//conditions:default": [],
    }),
    main_class = _TURBINE_MAIN_CLASS,
    profiles = select({
        ":turbine_direct_graal_do_pgo_instrument": [],
        "//conditions:default": ["profile.iprof"],
    }),
    # This provides libz.a on Linux instead of the host system.
    static_zlib = "//third_party/zlib",
    deps = [":turbine_deps"],
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

filegroup(
    name = "srcs",
    srcs = glob(
        ["**/*.java"],
        allow_empty = True,
    ) + [
        "BUILD",
    ],
)

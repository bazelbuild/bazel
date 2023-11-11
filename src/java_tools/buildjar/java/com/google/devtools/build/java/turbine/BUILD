load("@rules_java//java:defs.bzl", "java_binary")
load("@rules_graalvm//graalvm:defs.bzl", "native_image")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src/java_tools/buildjar:buildjar_package_group"],
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

native_image(
    name = "turbine_direct_graal",
    executable_name = select({
        "@bazel_tools//src/conditions:windows": "%target%.exe",
        "//conditions:default": "%target%",
    }),
    extra_args = [
        # Workaround for https://github.com/oracle/graal/issues/4757.
        "-H:-UseContainerSupport",
        # Do not fall back to bundling a full JVM when native image compilation fails.
        "--no-fallback",
        # More verbose errors in case of compilation failures.
        "-H:+ReportExceptionStackTraces",
    ] + select({
        "@platforms//os:linux": [
            # Statically link zlib but not glibc.
            "-H:+StaticExecutableWithDynamicLibC",
        ],
        "//conditions:default": [],
    }),
    main_class = _TURBINE_MAIN_CLASS,
    # This provides libz.a on Linux instead of the host system.
    static_zlib = "//third_party/zlib",
    deps = [":turbine_deps"],
)

filegroup(
    name = "srcs",
    srcs = glob(
        ["**/*.java"],
        allow_empty = True,
    ) + [
        "BUILD",
    ],
    visibility = ["//src:__subpackages__"],
)

# The Starlark interpreter
# Open-sourced as part of Bazel.

load("@rules_java//java:defs.bzl", "java_binary")

package(default_visibility = ["//src:__subpackages__"])

jvm_flags = [
    # quiet warnings from com.google.protobuf.UnsafeUtil,
    # see: https://github.com/google/protobuf/issues/3781
    # and: https://github.com/bazelbuild/bazel/issues/5599
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    # ... but only on JDK >= 9
    "-XX:+IgnoreUnrecognizedVMOptions",
]


java_binary(
    name = "Starlark",
    srcs = ["Starlark.java"],
    main_class = "com.google.devtools.starlark.Starlark",
    visibility = ["//visibility:public"],
    jvm_flags = jvm_flags,
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
    ],
)

java_binary(
    name = "Benchmark",
    srcs = ["Benchmark.java"],
    jvm_flags = jvm_flags,
    main_class = "com.google.devtools.starlark.Benchmark",
    visibility = ["//visibility:public"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

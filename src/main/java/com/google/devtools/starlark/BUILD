# The Starlark interpreter
# Open-sourced as part of Bazel.

load("@rules_java//java:defs.bzl", "java_binary")
load("@rules_jmh//:defs.bzl", "jmh_java_benchmarks")

package(default_visibility = ["//src:__subpackages__"])

java_binary(
    name = "Starlark",
    srcs = ["Starlark.java"],
    jvm_flags = [
        # quiet warnings from com.google.protobuf.UnsafeUtil,
        # see: https://github.com/google/protobuf/issues/3781
        # and: https://github.com/bazelbuild/bazel/issues/5599
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        # ... but only on JDK >= 9
        "-XX:+IgnoreUnrecognizedVMOptions",
    ],
    main_class = "com.google.devtools.starlark.Starlark",
    visibility = ["//visibility:public"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
    ],
)

[
    jmh_java_benchmarks(
        name = "Bench_" + file.replace('.', '_'),
        srcs = ["Bench.java"],
        main_class = "com.google.devtools.starlark.Bench",
        deps = [
            "//src/main/java/com/google/devtools/build/lib:events",
            "//src/main/java/com/google/devtools/build/lib:packages-internal",
            "//third_party:guava",
        ],
        data = [file],
        args = ["src/main/java/com/google/devtools/starlark/" + file],
    )
    for file in glob(["benchmarks/*"])
]

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

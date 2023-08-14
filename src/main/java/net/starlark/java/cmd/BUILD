# The Starlark interpreter
# Open-sourced as part of Bazel.

load("@rules_java//java:defs.bzl", "java_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_binary(
    name = "starlark",
    srcs = ["Main.java"],
    jvm_flags = [
        # quiet warnings from com.google.protobuf.UnsafeUtil,
        # see: https://github.com/google/protobuf/issues/3781
        # and: https://github.com/bazelbuild/bazel/issues/5599
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        # ... but only on JDK >= 9
        "-XX:+IgnoreUnrecognizedVMOptions",
    ],
    main_class = "net.starlark.java.cmd.Main",
    visibility = ["//src/main/java/net/starlark/java:clients"],
    deps = [
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party:jsr305",
    ],
)

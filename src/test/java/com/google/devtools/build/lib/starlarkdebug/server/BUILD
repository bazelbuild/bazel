load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "utils",
    srcs = ["MockDebugClient.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/starlarkdebug/proto:starlark_debugging_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_test(
    name = "StarlarkDebugServerTest",
    size = "medium",
    srcs = ["StarlarkDebugServerTest.java"],
    tags = ["requires-network"],  # https://github.com/bazelbuild/bazel/issues/20743
    deps = [
        ":utils",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/starlarkdebug/proto:starlark_debugging_java_proto",
        "//src/main/java/com/google/devtools/build/lib/starlarkdebug/server",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//src/test/java/com/google/devtools/build/lib/events:testutil",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DebugServerTransportTest",
    srcs = ["DebugServerTransportTest.java"],
    tags = ["requires-network"],  # https://github.com/bazelbuild/bazel/issues/20743
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/starlarkdebug/proto:starlark_debugging_java_proto",
        "//src/main/java/com/google/devtools/build/lib/starlarkdebug/server",
        "//src/test/java/com/google/devtools/build/lib/events:testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DebuggerSerializationTest",
    srcs = ["DebuggerSerializationTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/starlarkdebug/proto:starlark_debugging_java_proto",
        "//src/main/java/com/google/devtools/build/lib/starlarkdebug/server",
        "//src/main/java/net/starlark/java/annot",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "StarlarkDebugIntegrationTest",
    size = "medium",
    srcs = ["StarlarkDebugIntegrationTest.java"],
    jvm_flags = ["-Djava.net.preferIPv6Addresses=true"],  # https://github.com/bazelbuild/bazel/issues/5206
    deps = [
        ":utils",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_functions",
        "//src/main/java/com/google/devtools/build/lib/starlarkdebug/module",
        "//src/main/java/com/google/devtools/build/lib/starlarkdebug/proto:starlark_debugging_java_proto",
        "//src/main/java/com/google/devtools/build/lib/starlarkdebug/server",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

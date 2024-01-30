load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "ModCommandTests_lib",
    srcs = glob(
        ["*.java"],
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod:common",
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod:inspection",
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod:module_extension",
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod/modcommand",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/util:maybe_complete_set",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//src/test/java/com/google/devtools/build/lib/bazel/bzlmod:util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ModCommandTests",
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":ModCommandTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)

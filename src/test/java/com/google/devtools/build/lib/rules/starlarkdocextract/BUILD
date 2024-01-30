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
    name = "StarlarkDocExtractTests_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:actions/binary_file_write_action",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/bazel/bzlmod:resolution_impl",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository:repository_options",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/starlark",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/rules/starlarkdocextract",
        "//src/main/java/com/google/devtools/build/lib/skyframe:precomputed_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:skyframe_cluster",
        "//src/main/java/com/google/devtools/build/lib/starlarkbuildapi/repository",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skydoc/rendering:rendering_util",
        "//src/main/java/com/google/devtools/build/skydoc/rendering/proto:stardoc_output_java_proto",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/bazel/bzlmod:util",
        "//src/test/java/com/google/devtools/build/lib/starlark/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_test(
    name = "StarlarkDocExtractTests",
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":StarlarkDocExtractTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)

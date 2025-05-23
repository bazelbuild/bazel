load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]) + [
        "//src/test/java/com/google/devtools/build/lib/rules/android:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/apple:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/config:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/cpp:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/filegroup:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/genquery:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/java:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/objc:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/platform:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/proto:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/python:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/repository:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/starlarkdocextract:srcs",
        "//src/test/java/com/google/devtools/build/lib/rules/test:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "rules_tests",
    srcs = glob(["*.java"]),
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:file_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:build_options",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/rules:alias",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/skyframe:aspect_key_creator",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_value",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/bazel/bzlmod:util",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

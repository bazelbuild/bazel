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
    visibility = ["//src/test/java/com/google/devtools/build/lib/rules:__pkg__"],
)

java_library(
    name = "ObjcRulesTests_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_data",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:execution_requirements",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/instrumented_files_info",
        "//src/main/java/com/google/devtools/build/lib/analysis/config:compilation_mode",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:utils",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:provider",
        "//src/main/java/com/google/devtools/build/lib/rules/apple",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/rules/objc",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_value",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/net/starlark/java/eval",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//src/test/java/com/google/devtools/build/lib/rules/cpp:cc-info",
        "//src/test/java/com/google/devtools/build/lib/starlark/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/junit",
        "//third_party/java/testparameterinjector",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "ObjcRulesTests",
    shard_count = 20,
    tags = [
        "no_windows",
        "not_run:arm",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":ObjcRulesTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)

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
    name = "TestRulesTests_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_input",
        "//src/main/java/com/google/devtools/build/lib/actions:execution_requirements",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/analysis:analysis_cluster",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis:run_environment_info",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/execution_info",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "TestRulesTests",
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":TestRulesTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)

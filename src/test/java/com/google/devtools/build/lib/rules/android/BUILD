load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src/test/java/com/google/devtools/build/lib:test_lib"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib/rules:__pkg__"],
)

java_test(
    name = "AndroidResourceSourceDirectoryTest",
    srcs = ["AndroidResourceSourceDirectoryTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:provider",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_value",
        "//src/main/java/net/starlark/java/eval",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "AndroidConfigurationTest",
    srcs = ["AndroidConfigurationTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/rules/android",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party/java/junit",
    ],
)

java_test(
    name = "AndroidStarlarkCommonDexMergerTest",
    srcs = ["AndroidStarlarkCommonDexMergerTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis/actions:spawn_action_template",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

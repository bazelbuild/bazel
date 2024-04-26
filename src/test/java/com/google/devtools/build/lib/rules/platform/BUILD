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

TESTUTIL_SRCS = ["ToolchainTestCase.java"]

java_library(
    name = "PlatformRulesTests_lib",
    srcs = glob(
        ["*.java"],
        exclude = TESTUTIL_SRCS,
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:config/config_matching_provider",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:utils",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_value",
        "//src/main/java/net/starlark/java/eval",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "testutil",
    srcs = TESTUTIL_SRCS,
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:config/toolchain_type_requirement",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/toolchains:registered_toolchains_value",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/skyframe:testutil",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PlatformRulesTests",
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":PlatformRulesTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)

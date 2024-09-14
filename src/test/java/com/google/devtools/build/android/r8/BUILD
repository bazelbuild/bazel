load("@rules_java//java:defs.bzl", "java_library", "java_test")
load("//tools/android:android_genrule.bzl", "android_genrule")

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/android:__pkg__"],
)

# Description:
#   Tests for the bazel D8 bridge code.
java_library(
    name = "tests",
    srcs = [
        "AllTests.java",
    ] + select({
        "//external:has_androidsdk": glob(
            ["*Test.java"],
            exclude = [
                "NoAndroidSdkStubTest.java",
                "AllTests.java",
            ],
        ),
        "//conditions:default": ["NoAndroidSdkStubTest.java"],
    }),
    deps = [
        "//src/test/java/com/google/devtools/build/lib/testutil:TestSuite",
        "//src/tools/android/java/com/google/devtools/build/android/r8",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/java:jcommander",
        "@android_gmaven_r8//jar",
        "@bazel_tools//tools/java/runfiles",
    ],
)

java_test(
    name = "AllTests",
    size = "small",
    data = [
        ":arithmetic",
        ":barray",
        ":naming001",
        ":testdata_lambda_desugared.jar",
        ":twosimpleclasses",
    ],
    jvm_flags = [
        "-DCompatDexBuilderTests.twosimpleclasses=$(location :twosimpleclasses)",
        "-DCompatDexBuilderTests.naming001=$(location :naming001)",
        "-DCompatDxTests.arithmetic=$(location :arithmetic)",
        "-DCompatDxTests.barray=$(location :barray)",
        "-DCompatDexBuilderTests.lambda=$(location :testdata_lambda_desugared.jar)",
    ],
    runtime_deps = [
        ":tests",
    ],
)

java_library(
    name = "twosimpleclasses",
    srcs = glob(["testdata/twosimpleclasses/*.java"]),
    # Must be compiled as java8, otherwise CompatDx complains about invoke-customs
    javacopts = [
        "-source 8",
        "-target 8",
    ],
)

java_library(
    name = "naming001",
    srcs = glob(["testdata/naming001/*.java"]),
    # Must be compiled as java8, otherwise CompatDx complains about invoke-customs
    javacopts = [
        "-source 8",
        "-target 8",
    ],
)

java_library(
    name = "arithmetic",
    srcs = glob(["testdata/arithmetic/*.java"]),
    # Must be compiled as java8, otherwise CompatDx complains about invoke-customs
    javacopts = [
        "-source 8",
        "-target 8",
    ],
)

java_library(
    name = "barray",
    srcs = glob(["testdata/barray/*.java"]),
    # Must be compiled as java8, otherwise CompatDx complains about invoke-customs
    javacopts = [
        "-source 8",
        "-target 8",
    ],
)

java_library(
    name = "testdata_lambda",
    srcs = glob(["testdata/lambda/*.java"]),
)

# Needs to switch to dependencies to an android config.
android_genrule(
    name = "desugar_testdata_lambda",
    srcs = [
        ":testdata_lambda",
        "@bazel_tools//tools/android:android_jar",
    ],
    out = "testdata_lambda_desugared.jar",
    cmd = "$(location //src/tools/android/java/com/google/devtools/build/android/r8:desugar) " +
          "-i $(location :testdata_lambda) -o $@ " +
          "--bootclasspath_entry $(location @bazel_tools//tools/android:android_jar)",
    platform = "//tools/android:arm64-v8a",
    tools = ["//src/tools/android/java/com/google/devtools/build/android/r8:desugar"],
)

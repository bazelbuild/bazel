load("@rules_java//java:defs.bzl", "java_library", "java_test")

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
        "@bazel_tools//tools/java/runfiles",
    ] + select({
        "//external:has_androidsdk": [
            "//external:android/dx_jar_import",
            "//external:android/d8_jar_import",
        ],
        "//conditions:default": [],
    }),
)

java_test(
    name = "AllTests",
    size = "small",
    data = [
        ":arithmetic",
        ":barray",
        ":dexmergersample",
        ":naming001",
        ":twosimpleclasses",
    ],
    jvm_flags = [
        "-DCompatDexBuilderTests.twosimpleclasses=$(location :twosimpleclasses)",
        "-DCompatDexBuilderTests.naming001=$(location :naming001)",
        "-DCompatDxTests.arithmetic=$(location :arithmetic)",
        "-DCompatDxTests.barray=$(location :barray)",
        "-DDexFileMergerTest.dexmergersample=$(location :dexmergersample)",
    ],
    runtime_deps = [
        ":tests",
    ],
)

java_library(
    name = "twosimpleclasses",
    srcs = glob(["testdata/twosimpleclasses/*.java"]),
)

java_library(
    name = "naming001",
    srcs = glob(["testdata/naming001/*.java"]),
)

java_library(
    name = "arithmetic",
    srcs = glob(["testdata/arithmetic/*.java"]),
)

java_library(
    name = "barray",
    srcs = glob(["testdata/barray/*.java"]),
)

java_library(
    name = "dexmergersample",
    srcs = glob(["testdata/dexmergersample/*.java"]),
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
    visibility = ["//visibility:private"],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
    ],
    visibility = ["//src/test/java/com/google/devtools/build/android:__pkg__"],
)

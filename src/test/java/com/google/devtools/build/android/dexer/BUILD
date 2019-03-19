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
#   Tests for the blaze dx bridge code.
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
    resources = ["testresource.txt"],
    deps = [
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//src/tools/android/java/com/google/devtools/build/android/dexer",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:mockito2",
        "//third_party:truth",
        "@bazel_tools//tools/java/runfiles",
    ] + select({
        "//external:has_androidsdk": ["//external:android/dx_jar_import"],
        "//conditions:default": [],
    }),
)

java_library(
    name = "testdata",
    srcs = glob(["testdata/**/*.java"]),
)

java_test(
    name = "AllTests",
    size = "small",
    data = [
        "test_main_dex_list.txt",
        ":testdata",
        ":tests",
    ],
    jvm_flags = [
        "-Dtestmaindexlist=io_bazel/$(location :test_main_dex_list.txt)",
        "-Dtestinputjar=io_bazel/$(location :tests)",
        "-Dtestinputjar2=io_bazel/$(location :testdata)",
    ],
    runtime_deps = [
        ":tests",
    ],
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

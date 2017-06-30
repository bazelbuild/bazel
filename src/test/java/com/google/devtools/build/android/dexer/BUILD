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
        "//third_party:mockito",
        "//third_party:truth",
    ] + select({
        "//external:has_androidsdk": ["//external:android/dx_jar_import"],
        "//conditions:default": [],
    }),
)

java_test(
    name = "AllTests",
    data = [
        "test_main_dex_list.txt",
        ":tests",
    ],
    jvm_flags = [
        "-Dtestmaindexlist=$(location :test_main_dex_list.txt)",
        "-Dtestinputjar=$(location :tests)",
    ],
    runtime_deps = [
        ":tests",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/android:__pkg__"],
)

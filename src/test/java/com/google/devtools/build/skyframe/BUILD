package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

TESTUTIL_FILES = [
    "TrackingAwaiter.java",
    "GraphTester.java",
    "GenericFunctionException.java",
    "SomeErrorException.java",
    "TrackingProgressReceiver.java",
    "WalkableGraphUtils.java",
    # Truth Subject, SubjectFactory, and Graph files.
] + glob(["*Subject.java"]) + glob(["*SubjectFactory.java"]) + glob(["*Graph.java"]) + glob(["*Helper.java"])

java_library(
    name = "testutil",
    srcs = TESTUTIL_FILES,
    visibility = [
        "//src/test/java/com/google/devtools/build/lib:__subpackages__",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:truth",
    ],
)

java_test(
    name = "skyframe_base_test",
    srcs = glob(
        ["*.java"],
        exclude = TESTUTIL_FILES,
    ),
    test_class = "com.google.devtools.build.skyframe.AllTests",
    deps = [
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito2",
        "//third_party:truth",
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
    visibility = ["//src:__pkg__"],
)

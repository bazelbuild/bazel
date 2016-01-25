TESTUTIL_FILES = [
    "DeterministicInMemoryGraph.java",
    "NotifyingInMemoryGraph.java",
    "TrackingAwaiter.java",
    "GraphTester.java",
    "GenericFunctionException.java",
    "SomeErrorException.java",
    "TrackingInvalidationReceiver.java",
    "WalkableGraphUtils.java",
]

java_library(
    name = "testutil",
    srcs = TESTUTIL_FILES,
    visibility = [
        "//src/test/java/com/google/devtools/build/lib:__pkg__",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:collect",
        "//src/main/java/com/google/devtools/build/lib:concurrent",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/test/java/com/google/devtools/build/lib:testutil",
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
        "//src/main/java/com/google/devtools/build/lib:collect",
        "//src/main/java/com/google/devtools/build/lib:concurrent",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

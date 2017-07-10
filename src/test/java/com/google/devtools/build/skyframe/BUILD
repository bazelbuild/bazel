filegroup(
    name = "srcs",
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
        "//src/main/java/com/google/devtools/build/lib:collect",
        "//src/main/java/com/google/devtools/build/lib:concurrent",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
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
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

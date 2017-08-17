TEST_BASE_FILES = [
    "AbstractObjectCodecTest.java",
    "TestUtils.java",
]

java_library(
    name = "serialization-test-base",
    testonly = 1,
    srcs = TEST_BASE_FILES,
    visibility = [
        "//src/test/java/com/google/devtools/build/lib:__subpackages__",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "serialization-tests",
    testonly = 1,
    srcs = glob(
        [
            "*.java",
        ],
        exclude = TEST_BASE_FILES,
    ),
    deps = [
        ":serialization-test-base",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib:vfs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_test(
    name = "SerializationTests",
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":serialization-tests",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//src/test/java/com/google/devtools/build/lib/skyframe:__subpackages__",
    ],
)

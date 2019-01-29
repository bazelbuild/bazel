licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
    ),
    visibility = [
        "//src/test/java/com/google/devtools/build/lib:__pkg__",
    ],
)

java_test(
    name = "SkylarkCallableProcessorTest",
    srcs = ["SkylarkCallableProcessorTest.java"],
    resources = [":ProcessorTestFiles"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/skylarkinterface/processor:annotation_preprocessor",
        "//third_party:compile_testing",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

filegroup(
    name = "ProcessorTestFiles",
    srcs = glob(["testsources/*.java"]),
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
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)

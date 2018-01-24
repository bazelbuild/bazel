package(
    default_testonly = 1,
    default_visibility = ["//src/test/java/com/google/devtools/build/lib:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
)

java_test(
    name = "CallcountsTest",
    srcs = ["CallcountsTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/profiler/callcounts",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/pprof:profile_java_proto",
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
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)

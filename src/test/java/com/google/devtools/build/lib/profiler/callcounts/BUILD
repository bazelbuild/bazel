load("@rules_java//java:defs.bzl", "java_test")

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
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
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
    ],
)

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
)

java_test(
    name = "MainTest",
    srcs = ["MainTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/tools/benchmark/java/com/google/devtools/build/benchmark:benchmark_lib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "BazelBuildCaseTest",
    srcs = ["BazelBuildCaseTest.java"],
    deps = [
        "//src/tools/benchmark/java/com/google/devtools/build/benchmark:benchmark_lib",
        "//src/tools/benchmark/java/com/google/devtools/build/benchmark:build_data_java_proto",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "BazelBuilderTest",
    srcs = ["BazelBuilderTest.java"],
    # Needs git to run.
    tags = [
        "local",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:shell",
        "//src/tools/benchmark/java/com/google/devtools/build/benchmark:benchmark_lib",
        "//src/tools/benchmark/java/com/google/devtools/build/benchmark:build_data_java_proto",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

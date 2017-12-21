licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
    ),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_test(
    name = "SkylarkConfigurationFieldProcessorTest",
    srcs = ["SkylarkConfigurationFieldProcessorTest.java"],
    resources = [":ProcessorTestFiles"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib/analysis/skylark/annotations/processor:annotation_preprocessor",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//third_party:compile_testing",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

filegroup(
    name = "ProcessorTestFiles",
    srcs = glob(["optiontestsources/*.java"]),
)

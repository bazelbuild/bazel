filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
    ),
    visibility = ["//src/test/java/com/google/devtools/common/options:__pkg__"],
)

java_test(
    name = "OptionProcessorTest",
    srcs = ["OptionProcessorTest.java"],
    resources = [":OptionTestFiles"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options/processor:options_preprocessor",
        "//third_party:compile_testing",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

filegroup(
    name = "OptionTestFiles",
    srcs = glob(["optiontestsources/*.java"]),
)

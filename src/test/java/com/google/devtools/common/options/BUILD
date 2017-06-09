filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

java_test(
    name = "options_test",
    srcs = glob(["*.java"]),
    test_class = "com.google.devtools.common.options.AllTests",
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

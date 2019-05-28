package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)

java_test(
    name = "BazelShTestConfiguredTargetTest",
    srcs = ["BazelShTestConfiguredTargetTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/test/java/com/google/devtools/build/lib:analysis_testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

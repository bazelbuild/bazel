java_library(
    name = "deps",
    testonly = 1,
    exports = [
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "framework",
    testonly = 1,
    srcs = glob(["*.java"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib/blackbox:__subpackages__"],
    deps = [":deps"],
)

java_test(
    name = "framework_tests",
    testonly = 1,
    srcs = glob(["*.java"]),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [":deps"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib/blackbox:__pkg__"],
)

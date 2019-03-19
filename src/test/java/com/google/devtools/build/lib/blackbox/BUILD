package(
    default_testonly = 1,
    default_visibility = [
        "//javatests/com/google/devtools/build/lib:__subpackages__",
        "//src/test/java/com/google/devtools/build/lib:__subpackages__",
    ],
)

java_library(
    name = "base_deps",
    testonly = 1,
    visibility = ["//src/test/java/com/google/devtools/build/lib/blackbox:__subpackages__"],
    exports = [
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "test_deps",
    testonly = 1,
    visibility = ["//src/test/java/com/google/devtools/build/lib/blackbox:__subpackages__"],
    exports = [
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party:truth8",
    ],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/lib/blackbox/framework:srcs",
        "//src/test/java/com/google/devtools/build/lib/blackbox/bazel:srcs",
        "//src/test/java/com/google/devtools/build/lib/blackbox/tests:srcs",
        "//src/test/java/com/google/devtools/build/lib/blackbox/junit:srcs",
    ],
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)

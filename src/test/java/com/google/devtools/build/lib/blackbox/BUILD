load("@rules_java//java:defs.bzl", "java_library")

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
        "//src/main/java/com/google/devtools/build/lib/util:os",
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
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/lib/blackbox/bazel:srcs",
        "//src/test/java/com/google/devtools/build/lib/blackbox/framework:srcs",
        "//src/test/java/com/google/devtools/build/lib/blackbox/junit:srcs",
        "//src/test/java/com/google/devtools/build/lib/blackbox/tests:srcs",
    ],
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)

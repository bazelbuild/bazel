load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "EventsTests_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "EventsTests",
    size = "small",
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":EventsTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)

java_library(
    name = "testutil",
    testonly = 1,
    srcs = glob(["util/*.java"]) + ["EventTestTemplate.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//third_party:guava",
        "//third_party:junit4",
    ],
)

load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "PythonTests_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/rules/python:PythonTestUtils",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "PythonTests",
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":PythonTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)

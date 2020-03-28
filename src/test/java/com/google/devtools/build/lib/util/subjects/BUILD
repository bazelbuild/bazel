# Test-only package for Truth-compatible Subject and SubjectFactory types.
load("@rules_java//java:defs.bzl", "java_library")

package(
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
    name = "subjects",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib:exitcode-external",
        "//third_party:truth",
    ],
)

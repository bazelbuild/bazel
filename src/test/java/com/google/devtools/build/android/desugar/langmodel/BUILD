load("@rules_java//java:defs.bzl", "java_test")

# Description:
#   Tests for the Java 8 desugaring tool for Android.
package(
    default_testonly = 1,
    default_visibility = ["//src/test/java/com/google/devtools/build/android/desugar:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

java_test(
    name = "ClassMemberKeyTest",
    srcs = ["ClassMemberKeyTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/langmodel",
        "//third_party:asm",
        "//third_party:asm-tree",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ClassMemberRecordTest",
    srcs = ["ClassMemberRecordTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/langmodel",
        "//third_party:asm",
        "//third_party:asm-tree",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ClassMemberTrackReasonTest",
    srcs = ["ClassMemberTrackReasonTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/langmodel",
        "//third_party:asm",
        "//third_party:asm-tree",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

test_suite(
    name = "langmodel_test",
    tags = ["small"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
)

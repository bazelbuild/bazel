load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "common",
    srcs = glob(["*.java"]),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/remote/common",
        "//src/main/java/com/google/devtools/build/lib/remote/common:bulk_transfer_exception",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/junit",
        "//third_party/java/mockito",
        "//third_party/java/truth",
    ],
)

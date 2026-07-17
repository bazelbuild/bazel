load("@rules_java//java:defs.bzl", "java_library", "java_test")

# Description:
#   Tests for SingleJar
package(
    default_applicable_licenses = ["//:license"],
)

java_library(
    name = "libSingleJarTesting",
    testonly = 1,
    srcs = glob(["*.java"]),
    deps = [
        "//src/java_tools/singlejar/java/com/google/devtools/build/singlejar:libSingleJar",
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestSuite",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "SingleJarTests",
    test_class = "com.google.devtools.build.singlejar.SingleJarTests",
    runtime_deps = [":libSingleJarTesting"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*.java"]) + ["BUILD"],
    visibility = ["//src:__subpackages__"],
)

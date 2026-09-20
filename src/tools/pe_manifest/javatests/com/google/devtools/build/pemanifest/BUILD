load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/tools/pe_manifest:__pkg__"],
)

java_test(
    name = "PeManifestTest",
    size = "small",
    srcs = ["PeManifestTest.java"],
    data = [
        "//src/main/cpp:bazel.ico",
    ] + glob(["testdata/*"]),
    test_class = "com.google.devtools.build.pemanifest.PeManifestTest",
    deps = [
        "//src/tools/pe_manifest/java/com/google/devtools/build/pemanifest",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

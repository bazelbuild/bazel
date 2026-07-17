load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "UnsafeStringCodecTest",
    srcs = ["UnsafeStringCodecTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//third_party:junit4",
    ],
)

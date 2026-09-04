load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)

java_test(
    name = "StringUnsafeTest",
    srcs = ["StringUnsafeTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/unsafe:string",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

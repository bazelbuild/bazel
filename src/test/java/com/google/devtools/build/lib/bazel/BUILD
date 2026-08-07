load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]) + [
        "//src/test/java/com/google/devtools/build/lib/bazel/bzlmod:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/debug:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/repository:srcs",
        "//src/test/java/com/google/devtools/build/lib/bazel/rules:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "BazelWorkspaceStatusModuleTest",
    srcs = ["BazelWorkspaceStatusModuleTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bazel:modules",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

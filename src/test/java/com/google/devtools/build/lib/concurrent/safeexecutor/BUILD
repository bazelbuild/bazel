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
    name = "SafeExecutorTest",
    size = "medium",
    srcs = ["SafeExecutorTest.java"],
    tags = [
        "foundations",
    ],
    test_class = "com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutorTest",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent/safeexecutor:safe_executor",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

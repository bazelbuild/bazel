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
    name = "CallcountsTest",
    srcs = ["CallcountsTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/profiler/callcounts",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/pprof:profile_java_proto",
    ],
)

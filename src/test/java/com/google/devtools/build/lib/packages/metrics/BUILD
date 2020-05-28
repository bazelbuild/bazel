load("@rules_java//java:defs.bzl", "java_test")

licenses(["notice"])

package(
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
    name = "ExtremaPackageLoadingListenerTest",
    size = "small",
    srcs = ["ExtremaPackageLoadingListenerTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages/metrics",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

java_test(
    name = "PackageMetricsModuleTest",
    size = "small",
    srcs = ["PackageMetricsModuleTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/packages/metrics",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:junit4",
        "//third_party:mockito",
    ],
)

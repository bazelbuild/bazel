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
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)

java_test(
    name = "PackageMetricsPackageLoadingListenerTest",
    size = "small",
    srcs = ["PackageMetricsPackageLoadingListenerTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages/metrics",
        "//src/main/java/com/google/devtools/build/lib/packages/metrics:package_load_metrics_java_proto",
        "//src/main/java/com/google/devtools/build/lib/pkgcache:package_options",
        "//src/main/java/net/starlark/java/eval",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/mockito",
        "//third_party/java/protobuf/util:time",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "PackageMetricsModuleTest",
    size = "small",
    srcs = ["PackageMetricsModuleTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/packages/metrics",
        "//src/main/java/com/google/devtools/build/lib/runtime:blaze_command_cluster",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party/java/junit",
        "//third_party/java/mockito",
        "//third_party/java/truth",
    ],
)

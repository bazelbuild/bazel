load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = [
        "//src/test/java/com/google/devtools/build/lib:__pkg__",
        "//src/test/java/com/google/devtools/build/lib/skyframe:__pkg__",
    ],
)

java_library(
    name = "AbstractPackageLoaderTest",
    srcs = ["AbstractPackageLoaderTest.java"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/events:event_bus_event_handler",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/skyframe:bzl_load_failed_exception",
        "//src/main/java/com/google/devtools/build/lib/skyframe:skyframe_cluster",
        "//src/main/java/com/google/devtools/build/lib/skyframe/packages:AbstractPackageLoader",
        "//src/main/java/com/google/devtools/build/lib/skyframe/packages:PackageLoader",
        "//src/main/java/com/google/devtools/build/lib/util:value_or_exception",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "BazelPackageLoaderTest",
    srcs = ["BazelPackageLoaderTest.java"],
    deps = [
        ":AbstractPackageLoaderTest",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages/semantics",
        "//src/main/java/com/google/devtools/build/lib/skyframe/packages:BazelPackageLoader",
        "//src/main/java/com/google/devtools/build/lib/skyframe/packages:PackageLoader",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/net/starlark/java/eval",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

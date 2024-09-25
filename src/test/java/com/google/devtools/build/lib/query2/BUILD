load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]) + [
        "//src/test/java/com/google/devtools/build/lib/query2/aquery:srcs",
        "//src/test/java/com/google/devtools/build/lib/query2/common:srcs",
        "//src/test/java/com/google/devtools/build/lib/query2/cquery:srcs",
        "//src/test/java/com/google/devtools/build/lib/query2/engine:srcs",
        "//src/test/java/com/google/devtools/build/lib/query2/query/output:srcs",
        "//src/test/java/com/google/devtools/build/lib/query2/testutil:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "RBuildFilesVisitorTest",
    srcs = ["RBuildFilesVisitorTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/query2",
        "//src/main/java/com/google/devtools/build/lib/skyframe:directory_listing_state_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:directory_listing_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:package_lookup_value",
        "//src/main/java/com/google/devtools/build/lib/skyframe:sky_functions",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)

test_suite(
    name = "AllTests",
    tests = [
        "//src/test/java/com/google/devtools/build/lib/query2/cquery:BuildOutputFormatterCallbackTest",
        "//src/test/java/com/google/devtools/build/lib/query2/cquery:ProtoOutputFormatterCallbackTest",
        "//src/test/java/com/google/devtools/build/lib/query2/cquery:TransitionsOutputFormatterTest",
    ],
)

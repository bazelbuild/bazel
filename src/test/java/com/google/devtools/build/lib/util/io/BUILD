load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib/util:__pkg__"],
)

java_test(
    name = "io_tests",
    size = "large",
    srcs = glob(["*.java"]),
    tags = [
        "foundations",
        "no_windows",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/protobuf:bazel_flags_java_proto",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//third_party/java/guava:io",
        "//third_party/java/junit",
        "//third_party/java/mockito",
        "//third_party/java/protobuf",
        "//third_party/java/truth",
    ],
)

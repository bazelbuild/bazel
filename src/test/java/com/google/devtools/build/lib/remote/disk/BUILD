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
)

java_test(
    name = "disk",
    srcs = glob(["*.java"]),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/remote:store",
        "//src/main/java/com/google/devtools/build/lib/remote/common",
        "//src/main/java/com/google/devtools/build/lib/remote/common:action_key",
        "//src/main/java/com/google/devtools/build/lib/remote/common:cache_not_found_exception",
        "//src/main/java/com/google/devtools/build/lib/remote/disk",
        "//src/main/java/com/google/devtools/build/lib/remote/util:digest_util",
        "//src/main/java/com/google/devtools/build/lib/remote/util:futures",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/bazel",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//src/test/java/com/google/devtools/build/lib/testutil:external_file_system_lock",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:concurrent",
        "//third_party/java/junit",
        "//third_party/java/protobuf",
        "//third_party/java/protobuf:protobuf_lite",
        "//third_party/java/truth",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

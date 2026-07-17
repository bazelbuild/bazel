load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "VirtualCgroupTest",
    srcs = [
        "CpuTest.java",
        "HierarchyTest.java",
        "MemoryTest.java",
        "MountTest.java",
        "VirtualCgroupFactoryTest.java",
        "VirtualCgroupTest.java",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups",
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller",
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller:v1",
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller:v2",
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/proto:cgroups_info_java_proto",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

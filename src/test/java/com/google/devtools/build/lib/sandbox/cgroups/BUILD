load("@rules_java//java:defs.bzl", "java_library", "java_test")

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
    size = "small",
    srcs = ["VirtualCgroupTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "VirtualCgroupFactoryTest",
    size = "small",
    srcs = ["VirtualCgroupFactoryTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "MemoryTest",
    size = "small",
    srcs = ["MemoryTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller",
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller/v1",
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller/v2",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "CpuTest",
    size = "small",
    srcs = ["CpuTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller",
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller/v1",
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups/controller/v2",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "MountTest",
    size = "small",
    srcs = ["MountTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)


java_test(
    name = "HierarchyTest",
    size = "small",
    srcs = ["HierarchyTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/sandbox/cgroups",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

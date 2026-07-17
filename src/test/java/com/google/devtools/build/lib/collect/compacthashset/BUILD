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
    name = "CompactHashSetCodecTest",
    srcs = ["CompactHashSetCodecTest.java"],
    runtime_deps = [
        # Loaded reflectively.
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset:compacthashset_codec",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/testutils",
        "//third_party:junit4",
    ],
)

# Description:
#   one version C++ implementation.
package(default_visibility = ["//src:__subpackages__"])

cc_library(
    name = "duplicate_class_collector",
    srcs = [
        "duplicate_class_collector.cc",
    ],
    hdrs = [
        "duplicate_class_collector.h",
    ],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
    ],
)

cc_test(
    name = "duplicate_class_collector_test",
    srcs = [
        "duplicate_class_collector_test.cc",
    ],
    deps = [
        ":duplicate_class_collector",
        "@com_google_googletest//:gtest_main",
    ],
)

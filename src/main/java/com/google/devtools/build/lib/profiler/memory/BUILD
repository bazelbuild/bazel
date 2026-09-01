load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "allocationtracker_module",
    srcs = [
        "AllocationTrackerInstaller.java",
        "AllocationTrackerModule.java",
    ],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:build_lib",
    ],
    deps = [
        ":allocationtracker",
        ":current_rule_tracker",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_version_info",
        "//src/main/java/com/google/devtools/build/lib/analysis:server_directories",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/runtime:blaze_command_cluster",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
        "//third_party/java/allocation_instrumenter",
    ],
)

java_library(
    name = "current_rule_tracker",
    srcs = ["CurrentRuleTracker.java"],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:__subpackages__",
        "//src/test/java/com/google/devtools/build/lib/profiler/memory:__subpackages__",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party/java/guava:base",
    ],
)

java_library(
    name = "allocationtracker",
    srcs = [
        "AllocationTracker.java",
    ],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:__subpackages__",
        "//src/test/java/com/google/devtools/build/lib/profiler/memory:__subpackages__",
    ],
    deps = [
        ":current_rule_tracker",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party/java/allocation_instrumenter:sampler",
        "//third_party/java/caffeine",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
        "//third_party/pprof:profile_java_proto",
    ],
)

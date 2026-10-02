load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "allocation_tracker_module",
    srcs = ["AllocationTrackerModule.java"],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:build_lib",
    ],
    deps = [
        ":allocation_tracker",
        ":allocation_tracker_service",
        ":current_rule_tracker",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_version_info",
        "//src/main/java/com/google/devtools/build/lib/analysis:server_directories",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/runtime:blaze_command_cluster",
        "//src/main/java/com/google/devtools/build/lib/runtime:blaze_service",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
        "//third_party/java/guava:base",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "allocation_tracker_service",
    srcs = ["AllocationTrackerService.java"],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:build_lib",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/runtime:blaze_service",
        "//src/main/java/com/google/devtools/build/lib/skybridge:skybridge_interface",
    ],
)

java_library(
    name = "allocation_tracker_service_impl",
    srcs = ["AllocationTrackerServiceImpl.java"],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib/bazel:__pkg__",
    ],
    deps = [
        ":allocation_tracker_installer",
        ":allocation_tracker_service",
        "//src/main/java/com/google/devtools/build/lib/skybridge:sc_only",
    ],
)

java_library(
    name = "allocation_tracker_installer",
    srcs = ["AllocationTrackerInstaller.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skybridge:sc_only",
        "//third_party/java/allocation_instrumenter",
        "//third_party/java/allocation_instrumenter:sampler",
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
    name = "allocation_tracker",
    srcs = ["AllocationTracker.java"],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:__subpackages__",
        "//src/test/java/com/google/devtools/build/lib/profiler/memory:__subpackages__",
    ],
    deps = [
        ":allocation_tracker_service",
        ":current_rule_tracker",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party/java/caffeine",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
        "//third_party/pprof:profile_java_proto",
    ],
)

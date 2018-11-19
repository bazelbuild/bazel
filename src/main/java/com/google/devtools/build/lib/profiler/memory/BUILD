package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "allocationtracker_module",
    srcs = [
        "AllocationTrackerInstaller.java",
        "AllocationTrackerModule.java",
    ],
    deps = [
        ":allocationtracker",
        ":current_rule_tracker",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party/allocation_instrumenter",
    ],
)

java_library(
    name = "current_rule_tracker",
    srcs = ["CurrentRuleTracker.java"],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:__pkg__",
        "//src/test/java/com/google/devtools/build/lib/profiler/memory:__subpackages__",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//third_party:guava",
        "//third_party/pprof:profile_java_proto",
    ],
)

java_library(
    name = "allocationtracker",
    srcs = [
        "AllocationTracker.java",
    ],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:__pkg__",
        "//src/test/java/com/google/devtools/build/lib/profiler/memory:__subpackages__",
    ],
    deps = [
        ":current_rule_tracker",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/allocation_instrumenter",
        "//third_party/pprof:profile_java_proto",
    ],
)

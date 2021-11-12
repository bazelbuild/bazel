load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "sleep_prevention_module",
    srcs = ["SleepPreventionModule.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/jni",
        "//third_party:guava",
    ],
)

java_library(
    name = "system_suspension_module",
    srcs = [
        "SystemSuspensionModule.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/jni",
        "//third_party:flogger",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "memory_pressure_counter",
    srcs = [
        "MemoryPressureCounter.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/jni",
    ],
)

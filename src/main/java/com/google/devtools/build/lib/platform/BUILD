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
    name = "sleep_prevention_module",
    srcs = ["SleepPreventionModule.java"],
    deps = [
        ":platform_native_deps_service",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/runtime",
        "//third_party:guava",
    ],
)

java_library(
    name = "system_suspension_module",
    srcs = [
        "SystemSuspensionEvent.java",
        "SystemSuspensionModule.java",
    ],
    deps = [
        ":platform_native_deps_service",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/runtime",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "system_thermal_module",
    srcs = [
        "SystemThermalEvent.java",
        "SystemThermalModule.java",
    ],
    deps = [
        ":platform_native_deps_service",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/runtime",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "system_disk_space_module",
    srcs = [
        "SystemDiskSpaceEvent.java",
        "SystemDiskSpaceModule.java",
    ],
    deps = [
        ":platform_native_deps_service",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/runtime",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "system_load_advisory_module",
    srcs = [
        "SystemLoadAdvisoryEvent.java",
        "SystemLoadAdvisoryModule.java",
    ],
    deps = [
        ":platform_native_deps_service",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/runtime",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "system_memory_pressure_module",
    srcs = [
        "SystemMemoryPressureModule.java",
    ],
    deps = [
        ":platform_native_deps_service",
        ":system_memory_pressure_monitor",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/runtime",
        "//third_party:guava",
    ],
)

java_library(
    name = "system_memory_pressure_monitor",
    srcs = [
        "SystemMemoryPressureEvent.java",
        "SystemMemoryPressureMonitor.java",
    ],
    deps = [
        ":platform_native_deps_service",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//third_party:flogger",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "system_cpu_speed_module",
    srcs = [
        "SystemCPUSpeedEvent.java",
        "SystemCPUSpeedModule.java",
    ],
    deps = [
        ":platform_native_deps_service",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/runtime",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

# IJ
java_library(
    name = "platform_native_deps_service",
    srcs = [
        "PlatformNativeDepsService.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/runtime:blaze_service",
        "//third_party:error_prone_annotations",
    ],
)

# SC-only
java_library(
    name = "platform_native_deps_service_impl",
    srcs = [
        "PlatformNativeDepsServiceImpl.java",
    ],
    deps = [
        ":platform_native_deps_service",
        "//src/main/java/com/google/devtools/build/lib/jni",
    ],
)

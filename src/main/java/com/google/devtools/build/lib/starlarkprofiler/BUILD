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
    name = "cpu_profiler_module",
    srcs = ["CpuProfilerModule.java"],
    deps = [
        ":cpu_profiler_service",
        "//src/main/java/com/google/devtools/build/lib/runtime",
        "//src/main/java/com/google/devtools/build/lib/runtime:blaze_service",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
    ],
)

java_library(
    name = "cpu_profiler_service",
    srcs = ["CpuProfilerService.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/runtime:blaze_service",
        "//src/main/java/net/starlark/java/eval:cpu_profiler_native_support",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "cpu_profiler_service_impl",
    srcs = ["CpuProfilerServiceImpl.java"],
    deps = [
        ":cpu_profiler_service",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/net/starlark/java/eval:cpu_profiler_native_support",
        "//src/main/java/net/starlark/java/eval:cpu_profiler_native_support_impl",
        "//third_party:jsr305",
    ],
)

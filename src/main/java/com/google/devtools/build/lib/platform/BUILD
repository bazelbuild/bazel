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
        ":jni_loader",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party:guava",
    ],
)

java_library(
    name = "suspend_counter",
    srcs = [
        "SuspendCounter.java",
    ],
    deps = [
        ":jni_loader",
    ],
)

java_library(
    name = "memory_pressure_counter",
    srcs = [
        "MemoryPressureCounter.java",
    ],
    deps = [
        ":jni_loader",
    ],
)

java_library(
    name = "jni_loader",
    srcs = ["JniLoader.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/unix",
        "//src/main/java/com/google/devtools/build/lib/unix/jni",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/windows/jni:jni-loader",
    ],
)

load("@rules_java//java:defs.bzl", "java_test")

package(
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
    name = "SleepPreventionModuleTest",
    timeout = "short",
    srcs = ["SleepPreventionModuleTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/platform:sleep_prevention_module",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "SuspendCounterTest",
    timeout = "short",
    srcs = ["SuspendCounterTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/platform:suspend_counter",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:process",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "MemoryPressureCounterTest",
    timeout = "short",
    srcs = ["MemoryPressureCounterTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/platform:memory_pressure_counter",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

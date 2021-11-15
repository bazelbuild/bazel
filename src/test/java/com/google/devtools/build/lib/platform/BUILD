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
    name = "SystemSuspensionEventTest",
    timeout = "short",
    srcs = ["SystemSuspensionEventTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/platform:system_suspension_module",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//third_party:guava",
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

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/native:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = glob([
        "*.cc",
        "*.h",
    ]) + [
        "BUILD",
    ],
    visibility = ["//src/main/native:__pkg__"],
)

cc_library(
    name = "lib-file",
    srcs = [
        "file.cc",
        "util.cc",
    ],
    hdrs = [
        "file.h",
        "util.h",
    ],
    linkopts = [
        "-DEFAULTLIB:advapi32.lib",  # RegGetValueW
    ],
    visibility = [
        "//src/main/cpp:__subpackages__",
        "//src/main/tools:__pkg__",
        "//src/test/cpp:__subpackages__",
        "//src/test/native:__subpackages__",
        "//src/tools/launcher/util:__pkg__",
        "//tools/test:__pkg__",
    ],
)

cc_library(
    name = "lib-process",
    srcs = ["process.cc"],
    hdrs = ["process.h"],
    visibility = [
        "//src/main/cpp:__pkg__",
        "//src/test/native:__subpackages__",
        "//src/tools/launcher:__pkg__",
        "//tools/test:__pkg__",
    ],
    deps = [":lib-file"],
)

cc_binary(
    name = "windows_jni.dll",
    srcs = [
        "file-jni.cc",
        "jni-util.cc",
        "jni-util.h",
        "processes-jni.cc",
        "sleep_prevention_jni.cc",
        "system_cpu_speed_monitor_jni.cc",
        "system_disk_space_monitor_jni.cc",
        "system_load_advisory_monitor_jni.cc",
        "system_memory_pressure_jni.cc",
        "system_network_stats.cc",
        "system_suspension_monitor_jni.cc",
        "system_thermal_monitor_jni.cc",
        "//src/main/native:jni.h",
        "//src/main/native:jni_md.h",
    ],
    linkshared = 1,
    visibility = [
        "//src/main/java/com/google/devtools/build/lib/jni:__pkg__",
    ],
    deps = [
        ":lib-file",
        ":lib-process",
        "//src/main/native:blake3_jni",
    ],
)

filegroup(
    name = "embedded_java_tools",
    srcs = [
        "file.cc",
        "file.h",
        "util.cc",
        "util.h",
    ],
    visibility = ["//src/main/cpp/util:__pkg__"],
)

# A zip containing source files used by ijar.
genrule(
    name = "native_windows_zip",
    srcs = [":embedded_java_tools"],
    outs = ["native_windows.zip"],
    cmd = "$(location //src:zip_files) src/main/native/windows $@ $(SRCS)",
    tools = ["//src:zip_files"],
    visibility = ["//src/main/cpp/util:__pkg__"],
)

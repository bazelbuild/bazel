load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "unix_file_system",
    srcs = [
        "InvalidArgumentIOException.java",
        "UnixFileStatus.java",
        "UnixFileSystem.java",
    ],
    deps = [
        ":native_posix_files_service",
        ":unix_mode",
        "//src/main/java/com/google/devtools/build/lib/concurrent:thread_safety",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/util:blocker",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "unix_mode",
    srcs = ["UnixMode.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/vfs",
    ],
)

java_library(
    name = "native_posix_files_service",
    srcs = [
        "InvalidArgumentIOException.java",
        "NativePosixFilesService.java",
    ],
    deps = ["//src/main/java/com/google/devtools/build/lib/runtime:blaze_service"],
)

java_library(
    name = "native_posix_files_service_impl",
    srcs = ["NativePosixFilesServiceImpl.java"],
    deps = [
        ":native_posix_files_service",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/jni",
    ],
)

java_library(
    name = "procmeminfo_parser",
    srcs = ["ProcMeminfoParser.java"],
    deps = ["//third_party:guava"],
)

java_library(
    name = "process_utils_service",
    srcs = ["ProcessUtilsService.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/runtime:blaze_service",
    ],
)

java_library(
    name = "process_utils_service_impl",
    srcs = ["ProcessUtilsServiceImpl.java"],
    deps = [
        ":process_utils_service",
        "//src/main/java/com/google/devtools/build/lib/jni",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

# Low level zip archive processing library.

load("@rules_java//java:defs.bzl", "java_binary", "java_library")

package(
    default_visibility = [
        "//src/test/java/com/google/devtools/build/android/ziputils:__pkg__",
        "//src/tools/android/java/com/google/devtools/build/android:__subpackages__",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

filegroup(
    name = "embedded_tools",
    srcs = ["BUILD.tools"],
)

java_library(
    name = "ziputils_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_options_utils",
        "//third_party:guava",
        "//third_party:jsr305",
        "@maven//:com_beust_jcommander",
    ],
)

java_binary(
    name = "mapper",
    main_class = "com.google.devtools.build.android.ziputils.DexMapper",
    visibility = ["//visibility:private"],
    runtime_deps = [":ziputils_lib"],
)

java_binary(
    name = "reducer",
    main_class = "com.google.devtools.build.android.ziputils.DexReducer",
    visibility = ["//visibility:private"],
    runtime_deps = [":ziputils_lib"],
)

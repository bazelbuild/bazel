# Low level zip archive processing library.

package(
    default_visibility = [
        "//src/test/java/com/google/devtools/build/android/ziputils:__pkg__",
        "//src/tools/android/java/com/google/devtools/build/android:__pkg__",
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
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
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

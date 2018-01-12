package(default_visibility = ["//visibility:private"])

package_group(
    name = "android-prod",
    packages = [
        "//src/tools/android/java/com/google/devtools/build/android",
    ],
)

package_group(
    name = "android-tests",
    packages = [
        "//src/test/java/com/google/devtools/build/android/...",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [":android-prod"],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["*.java"]),
    visibility = [":android-prod"],
)

java_library(
    name = "junctions",
    srcs = glob(["*.java"]),
    data = select({
        "//src/conditions:windows": ["//src/main/native/windows:windows_jni.dll"],
        "//conditions:default": [],
    }),
    visibility = [
        ":android-prod",
        ":android-tests",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/windows/jni:file",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

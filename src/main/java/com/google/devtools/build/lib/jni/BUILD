load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = ["//src:__subpackages__"],
)

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "jni",
    srcs = ["JniLoader.java"],
    resources = select({
        "//src/conditions:darwin": [
            "//src/main/native:libunix_jni.dylib",
        ],
        "//src/conditions:windows": [
            "//src/main/native/windows:windows_jni.dll",
        ],
        "//conditions:default": [
            "//src/main/native:libunix_jni.so",
        ],
    }),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

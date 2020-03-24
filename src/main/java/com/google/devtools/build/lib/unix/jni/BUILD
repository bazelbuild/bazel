load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib/unix:__pkg__"],
)

# Java interfaces to various native UNIX system services.
#
# All programs that depend on libraries in this package will have
# JNI (native code) dependencies.
java_library(
    name = "jni",
    srcs = [
        "UnixJniLoader.java",
    ],
    data = [
        "//src/main/native:target-os-unix-native-lib",
    ],
)

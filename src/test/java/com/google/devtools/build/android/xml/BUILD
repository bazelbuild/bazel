load("@rules_java//java:defs.bzl", "java_test")

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/android:__pkg__"],
)

java_test(
    name = "IdXmlResourceValueTest",
    size = "small",
    srcs = ["IdXmlResourceValueTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//src/tools/android/java/com/google/devtools/build/android/proto:resources_java_proto",
        "//src/tools/android/java/com/google/devtools/build/android/resources",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ProtoXmlUtilsTest",
    size = "small",
    srcs = ["ProtoXmlUtilsTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//src/tools/android/java/com/google/devtools/build/android/proto:resources_java_proto",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "SimpleXmlResourceValueTest",
    size = "small",
    srcs = ["SimpleXmlResourceValueTest.java"],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//src/tools/android/java/com/google/devtools/build/android/proto:resources_java_proto",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

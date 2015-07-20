android_library(
    name = "lib",
    srcs = ["Lib.java"],
)

android_binary(
    name = "hello_world",
    srcs = glob([
        "MainActivity.java",
        "Jni.java",
    ]),
    manifest = "AndroidManifest.xml",
    deps = [
        ":jni",
        ":lib",
    ],
)

cc_library(
    name = "jni",
    srcs = ["jni.cc"],
)

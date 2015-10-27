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
    legacy_native_support = 0,
    manifest = "AndroidManifest.xml",
    deps = [
        ":jni",
        ":lib",
    ],
)

cc_library(
    name = "jni",
    srcs = ["jni.cc"],
    deps = [":jni_dep"],
)

cc_library(
    name = "jni_dep",
    srcs = ["jni_dep.cc"],
    hdrs = ["jni_dep.h"],
)

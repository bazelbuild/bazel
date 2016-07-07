filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/test/java/com/google/devtools/build/android/ideinfo:srcs",
        "//src/test/java/com/google/devtools/build/android/idlclass:srcs",
        "//src/test/java/com/google/devtools/build/android/resources:srcs",
        "//src/test/java/com/google/devtools/build/android/ziputils:srcs",
    ],
    visibility = ["//src:__pkg__"],
)

java_test(
    name = "RClassGeneratorActionTest",
    size = "medium",
    srcs = glob(["*.java"]),
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_builder_lib",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

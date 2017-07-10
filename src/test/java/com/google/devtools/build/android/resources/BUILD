filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/android:__pkg__"],
)

java_test(
    name = "RClassGeneratorTest",
    size = "small",
    srcs = glob(["*.java"]),
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/resources",
        "//third_party:android_common_25_0_0",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

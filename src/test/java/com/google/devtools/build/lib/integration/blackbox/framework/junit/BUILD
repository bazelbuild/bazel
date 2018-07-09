java_test(
    name = "TimeoutTestWatcherTest",
    srcs = [
        "TimeoutTestWatcher.java",
        "TimeoutTestWatcherBaseTest.java",
        "TimeoutTestWatcherTest.java",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "timeout_watcher",
    srcs = ["TimeoutTestWatcher.java"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party:junit4",
    ],
)

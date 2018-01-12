package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/android:__pkg__"],
)

java_test(
    name = "JunctionCreatorTest",
    size = "small",
    srcs = select({
        "//src/conditions:windows": ["WindowsJunctionCreatorTest.java"],
        "//conditions:default": ["NoopJunctionCreatorTest.java"],
    }),
    test_class = select({
        "//src/conditions:windows": "com.google.devtools.build.android.junctions.WindowsJunctionCreatorTest",
        "//conditions:default": "com.google.devtools.build.android.junctions.NoopJunctionCreatorTest",
    }),
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/junctions",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
    visibility = ["//visibility:private"],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
    ],
    visibility = ["//src/test/java/com/google/devtools/build/android:__pkg__"],
)

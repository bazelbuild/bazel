package(default_visibility = ["//visibility:private"])

package_group(
    name = "android-tests",
    packages = [
        "//src/test/java/com/google/devtools/build/android/...",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [":android-tests"],
)

java_test(
    name = "JunctionCreatorTest",
    size = "small",
    srcs = select({
        "//src:windows": ["WindowsJunctionCreatorTest.java"],
        "//src:windows_msvc": ["WindowsJunctionCreatorTest.java"],
        "//src:windows_msys": ["WindowsJunctionCreatorTest.java"],
        "//conditions:default": ["NoopJunctionCreatorTest.java"],
    }),
    test_class = select({
        "//src:windows": "com.google.devtools.build.android.junctions.WindowsJunctionCreatorTest",
        "//src:windows_msvc": "com.google.devtools.build.android.junctions.WindowsJunctionCreatorTest",
        "//src:windows_msys": "com.google.devtools.build.android.junctions.WindowsJunctionCreatorTest",
        "//conditions:default": "com.google.devtools.build.android.junctions.NoopJunctionCreatorTest",
    }),
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/junctions",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

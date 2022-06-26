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

filegroup(
    name = "test_data",
    srcs = [
        "brokenManifest/AndroidManifest.xml",
        "expected-merged-permissions/AndroidManifest.xml",
        "expected/AndroidManifest.xml",
        "mergeeOne/AndroidManifest.xml",
        "mergeeTwo/AndroidManifest.xml",
        "merger/AndroidManifest.xml",
    ],
)

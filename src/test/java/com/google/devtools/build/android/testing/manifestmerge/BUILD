package(default_visibility = ["//src/test/java/com/google/devtools/build/android:__pkg__"])

filegroup(
    name = "test_data",
    srcs = [
        "expected/AndroidManifest.xml",
        "mergeeOne/AndroidManifest.xml",
        "mergeeTwo/AndroidManifest.xml",
        "merger/AndroidManifest.xml",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

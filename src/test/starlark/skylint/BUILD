package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

py_test(
    name = "skylint_test",
    srcs = [
        "skylint_test.py",
        "testenv.py",
    ],
    data = ["//src/tools/skylark/java/com/google/devtools/skylark/skylint:Skylint"] + glob([
        "testdata/*",
    ]),
)

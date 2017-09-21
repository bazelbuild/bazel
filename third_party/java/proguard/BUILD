licenses(["restricted"])  # GNU GPL v2

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//third_party:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "proguard5.3.3/docs/GPL.html",
        "proguard5.3.3/lib/proguard.jar",
    ],
    visibility = ["//src:__pkg__"],
)

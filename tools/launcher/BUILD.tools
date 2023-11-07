package(default_visibility = ["//visibility:public"])

filegroup(
    name = "launcher_windows",
    srcs = select({
        "//src/conditions:remote": ["//src/tools/launcher:launcher"],
        "//conditions:default": ["launcher.exe"],
    }),
)

filegroup(
    name = "launcher",
    srcs = select({
        "//src/conditions:host_windows": [":launcher_windows"],
        "//conditions:default": [
            "//src/tools/launcher:launcher",
        ],
    }),
)

filegroup(
    name = "launcher_maker",
    srcs = select({
        "//src/conditions:host_windows": [":launcher_maker_windows"],
        "//conditions:default": [
            "//src/tools/launcher:launcher_maker",
        ],
    }),
)

filegroup(
    name = "launcher_maker_windows",
    srcs = select({
        "//src/conditions:remote": ["//src/tools/launcher:launcher_maker"],
        "//conditions:default": ["launcher_maker.exe"],
    }),
)

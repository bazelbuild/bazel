package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

filegroup(
    name = "prebuilt_singlejar",
    srcs = select({
        ":windows": ["java_tools/src/tools/singlejar/singlejar_local.exe"],
        "//conditions:default": ["java_tools/src/tools/singlejar/singlejar_local"],
    }),
)

filegroup(
    name = "ijar_prebuilt_binary",
    srcs = select({
        ":windows": ["java_tools/ijar/ijar.exe"],
        "//conditions:default": ["java_tools/ijar/ijar"],
    }),
)

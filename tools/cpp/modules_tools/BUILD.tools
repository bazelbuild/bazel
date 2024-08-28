package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

filegroup(
    name = "aggregate-ddi-bin",
    srcs = select({
        ":windows": ["aggregate-ddi.exe"],
        "//conditions:default": ["aggregate-ddi"],
    }),
)

filegroup(
    name = "generate-modmap-bin",
    srcs = select({
        ":windows": ["generate-modmap.exe"],
        "//conditions:default": ["generate-modmap"],
    }),
)

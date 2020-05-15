load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
string_flag(
    name = "myflag",
    values = ["a", "b", "c"],
    build_setting_default = "a",
    )

config_setting(
    name = "match",
    flag_values = {":myflag": "c"})
    
filegroup(
    name = "fg",
    srcs = select({
      ":match": ["fg.txt"],
    })
)

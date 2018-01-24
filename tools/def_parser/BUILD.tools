package(default_visibility = ["//visibility:public"])

filegroup(
    name = "def_parser_windows",
    srcs = select({
      "//src/conditions:remote": ["//third_party/def_parser:def_parser"],
      "//conditions:default": ["def_parser.exe"],
    }),
)

filegroup(
    name = "def_parser",
    srcs = select({
      "//src/conditions:host_windows": ["def_parser_windows"],
      "//conditions:default": [
        "no_op.bat",
      ],
    }),
)

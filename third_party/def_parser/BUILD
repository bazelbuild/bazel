licenses(["notice"])  # 3-clause BSD

cc_library(
  name = "def_parser_lib",
  srcs = ["def_parser.cc"],
  hdrs = ["def_parser.h"],
  deps = [
    "//src/main/cpp/util:file",
  ],
)

cc_binary(
  name = "def_parser",
  srcs = ["def_parser_main.cc"],
  deps = [
    ":def_parser_lib",
  ],
  visibility = [
    "//src:__pkg__",
  ],
)

py_test(
  name = "def_parser_test",
  srcs = ["def_parser_test.py"],
  data = [":def_parser"],
  deps = ["//src/test/py/bazel:test_base"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
      "//src:__pkg__",
      "//third_party:__pkg__",
    ],
)

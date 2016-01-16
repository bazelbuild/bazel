package(default_visibility = ["//visibility:public"])

load("/tools/build_rules/go/def", "go_prefix", "go_library", "go_binary", "go_test")

go_library(
    name = "filter_tags_lib",
    srcs = ["filter_tags.go"],
)

go_binary(
    name = "filter_tags",
    library = ":filter_tags_lib",
)

go_test(
    name = "filter_tags_test",
    srcs = ["filter_tags_test.go"],
    library = ":filter_tags_lib",
)

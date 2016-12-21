licenses(["notice"])  # MIT, fonts under OFL 1.1

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//third_party:__pkg__"],
)

filegroup(
    name = "font_awesome_css",
    srcs = [
        "css/font-awesome.css",
        "css/font-awesome.min.css",
    ],
    visibility = ["//site:__pkg__"],
)

filegroup(
    name = "font_awesome_font",
    srcs = glob(["fonts/fontawesome-webfont.*"]),
    visibility = ["//site:__pkg__"],
)

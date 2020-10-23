package(default_visibility = ["//scripts/packages:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

genrule(
    name = "bazel-dmg",
    srcs = [
        "//scripts/packages/dmg:resource/bazel-icon.rsrc",
        "//scripts/packages/dmg:resource/background.png",
        "//scripts/packages/dmg:resource/DS_Store",
        "//scripts/packages:with-jdk/bazel-real",
    ],
    outs = [
        "bazel.dmg",
    ],
    cmd = " ".join([
        "$(location //scripts/packages/dmg:create_bazel_dmg.sh)",
        "--bazel_binary=$(location //scripts/packages:with-jdk/bazel-real)",
        "--output=$@",
        "--bazel_icon=$(location //scripts/packages/dmg:resource/bazel-icon.rsrc)",
        "--ds_store=$(location //scripts/packages/dmg:resource/DS_Store)",
        "--background=$(location //scripts/packages/dmg:resource/background.png)",
    ]),
    tags = ["no-sandbox"],
    tools = ["//scripts/packages/dmg:create_bazel_dmg.sh"],
)

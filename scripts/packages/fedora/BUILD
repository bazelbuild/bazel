package(default_visibility = ["//scripts/packages:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

genrule(
    name = "bazel-fedora",
    srcs = [
        "bazel.spec",
        "//scripts/packages:bazel",
        "//scripts/packages:bazel-real",
        "//scripts/packages:debian/bazel.bazelrc",
        "//scripts:bash_completion",
    ],
    outs = [
        "bazel-fedora.rpm",
    ],
    cmd = """
      $(location :build_rpm.sh) \
        $(location :bazel.spec) \
        $(location bazel-fedora.rpm) \
        $(location //scripts/packages:bazel) \
        $(location //scripts/packages:bazel-real) \
        $(location //scripts/packages:debian/bazel.bazelrc) \
        $(location //scripts:bash_completion)
    """,
    tools = ["build_rpm.sh"],
)

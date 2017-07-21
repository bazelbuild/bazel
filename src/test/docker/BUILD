load("//tools/build_defs/docker:docker.bzl", "docker_build")
load("//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("//src/test/docker:flavours.bzl", "FLAVOURS")

pkg_tar(
    name = "bazel_cc_configure",
    package_dir = "/opt/workspace",
    strip_prefix = "/",
    deps = ["//:bazel-distfile-tar"],
)

[
    [docker_build(
        name = "bazel_cc_configure-%s-%s" % (flavour, mode),
        base = "@docker-%s//:image" % flavour,
        entrypoint = "/opt/workspace/compile.sh && ./output/bazel test ${EXTRA_BAZEL_ARGS} //examples/cpp:hello-success_test",
        env = {
            "EXTRA_BAZEL_ARGS": "--spawn_strategy=standalone --genrule_strategy=standalone -c %s" % mode,
        },
        tars = [":bazel_cc_configure"],
        workdir = "/opt/workspace",
    ) for mode in [
        "dbg",
        "opt",
        "fastbuild",
    ]]
    for flavour in FLAVOURS
]

# TODO(#2418): reactivate debian-stretch opt (flaky)
[
    [py_test(
        name = "test_cc_configure-%s-%s" % (flavour, mode),
        size = "large",
        srcs = ["docker_test.py"],
        args = [
            "--main='$(location :bazel_cc_configure-%s-%s)'" % (flavour, mode),
            "--docker='../docker/docker'",
        ],
        data = [
            ":bazel_cc_configure-%s-%s" % (flavour, mode),
            "@docker//:docker",
        ],
        local = 1,
        main = "docker_test.py",
        tags = [
            "docker",
            "exclusive",
            "local",
        ],
        deps = ["//third_party/py/gflags"],
    ) for mode in [
        "dbg",
        "opt",
        "fastbuild",
    ]]
    for flavour in FLAVOURS
    if flavour != "debian-stretch"
]

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

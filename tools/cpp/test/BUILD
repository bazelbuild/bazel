load("//tools/build_defs/docker:docker.bzl", "docker_build")
load("//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

# This is totally non hermetic, we should really replace that by a
# docker_pull rule.
FLAVOURS = [f[f.find(".") + 1:] for f in glob(["Dockerfile.*"])]

[
    # This is totally non hermetic.
    genrule(
        name = "docker-" + flavour,
        srcs = ["Dockerfile." + flavour],
        outs = ["docker-%s.tar" % flavour],
        cmd = "\n".join([
            "DIR=\"$$(dirname $<)\"",
            "IMG=\"$$(basename $<)\"",
            "DOCKER=\"$${PWD}/$(location @docker//:docker)\"",
            "(",
            "   cd $$DIR",
            "   $$DOCKER build -f $$IMG -t bazel_tools_cpp_test:%s ." % flavour,
            ")",
            "$$DOCKER save bazel_tools_cpp_test:%s > $@" % flavour,
        ]),
        tags = ["local"],
        # Docker needs to knows how to contact the daemon from the environment.
        # @docker//:docker point to a docker wrapper that copy the environment
        # of the user.
        tools = ["@docker//:docker"],
    )
    for flavour in FLAVOURS
]

# Just to avoid re-reading docker images all the time
[
    docker_build(
        name = "base-" + flavour,
        base = "docker-" + flavour,
    )
    for flavour in FLAVOURS
]

pkg_tar(
    name = "bazel_cc_configure",
    package_dir = "/opt/workspace",
    strip_prefix = "/",
    deps = ["//:bazel-srcs"],
)

[
    [docker_build(
        name = "bazel_cc_configure-%s-%s" % (flavour, mode),
        base = ":base-" + flavour,
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
]

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/cpp:__pkg__"],
)

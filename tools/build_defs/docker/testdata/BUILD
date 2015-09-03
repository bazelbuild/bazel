package(
    default_visibility = [
        "//tools/build_defs/docker:__subpackages__",
    ],
)

load("/tools/build_defs/docker/docker", "docker_build")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

filegroup(
    name = "archive_testdata",
    srcs = glob(["archive/**"]),
)

genrule(
    name = "gen",
    outs = ["gen.out"],
    cmd = "echo generated > $@",
)

docker_build(
    name = "gen_image",
    files = [":gen"],
)

docker_build(
    name = "files_base",
    files = ["foo"],
)

docker_build(
    name = "files_with_files_base",
    base = ":files_base",
    files = ["bar"],
)

docker_build(
    name = "tar_base",
    tars = ["one.tar"],
)

docker_build(
    name = "tar_with_tar_base",
    base = ":tar_base",
    tars = ["two.tar"],
)

docker_build(
    name = "files_with_tar_base",
    base = ":tar_base",
    files = ["bar"],
)

docker_build(
    name = "tar_with_files_base",
    base = ":files_base",
    tars = ["two.tar"],
)

# TODO(mattmoor): Test scalar entrypoint
docker_build(
    name = "base_with_entrypoint",
    entrypoint = ["/bar"],
    files = ["bar"],
    ports = ["8080"],
    tars = ["two.tar"],
)

# TODO(mattmoor): Test scalar cmd
docker_build(
    name = "derivative_with_shadowed_cmd",
    base = ":base_with_entrypoint",
    cmd = ["shadowed-arg"],
    files = ["foo"],
)

docker_build(
    name = "derivative_with_cmd",
    base = ":derivative_with_shadowed_cmd",
    cmd = [
        "arg1",
        "arg2",
    ],
    ports = ["80/tcp"],
    tars = ["one.tar"],
)

docker_build(
    name = "base_with_volume",
    files = [
        "bar",
        "foo",
    ],
    volumes = ["/logs"],
)

docker_build(
    name = "derivative_with_volume",
    base = ":base_with_volume",
    volumes = [
        "/asdf",
        "/blah",
    ],
)

py_binary(
    name = "extras_gen",
    srcs = ["extras_gen.py"],
    deps = ["//tools/build_defs/docker:archive"],
)

genrule(
    name = "extras",
    outs = ["extras.tar"],
    cmd = "$(location :extras_gen) $@",
    tools = [":extras_gen"],
)

docker_build(
    name = "generated_tarball",
    tars = [
        ":extras",
    ],
)

docker_build(
    name = "with_env",
    base = ":base_with_volume",
    env = {
        "foo": "/asdf",
        "bar": "blah blah blah",
    },
)

docker_build(
    name = "with_double_env",
    base = ":with_env",
    env = {
        "baz": "${foo} $bar",
    },
)

docker_build(
    name = "link_with_files_base",
    base = ":files_base",
    symlinks = {
        "/usr/bin/java": "/bar",
    },
)

# Trying with the 'top' file
py_binary(
    name = "strip_top",
    srcs = ["strip_top.py"],
    deps = ["//tools/build_defs/docker:archive"],
)

[genrule(
    name = "notop_" + n,
    srcs = [":" + n + ".tar"],
    outs = ["notop_%s.tar" % n],
    cmd = "$(location :strip_top) $< $@",
    tools = [":strip_top"],
) for n in [
    "files_base",
    "tar_base",
    "base_with_entrypoint",
    "base_with_volume",
    "generated_tarball",
]]

docker_build(
    name = "notop_files_with_files_base",
    base = ":notop_files_base",
    files = ["bar"],
)

docker_build(
    name = "notop_tar_with_tar_base",
    base = ":notop_tar_base",
    tars = ["two.tar"],
)

docker_build(
    name = "notop_files_with_tar_base",
    base = ":notop_tar_base",
    files = ["bar"],
)

docker_build(
    name = "notop_tar_with_files_base",
    base = ":notop_files_base",
    tars = ["two.tar"],
)

docker_build(
    name = "notop_derivative_with_shadowed_cmd",
    base = ":notop_base_with_entrypoint",
    cmd = ["shadowed-arg"],
    files = ["foo"],
)

docker_build(
    name = "notop_derivative_with_cmd",
    base = ":notop_derivative_with_shadowed_cmd",
    cmd = [
        "arg1",
        "arg2",
    ],
    ports = ["80/tcp"],
    tars = ["one.tar"],
)

docker_build(
    name = "notop_derivative_with_volume",
    base = ":notop_base_with_volume",
    volumes = [
        "/asdf",
        "/blah",
    ],
)

docker_build(
    name = "notop_with_env",
    base = ":notop_base_with_volume",
    env = {
        "foo": "/asdf",
        "bar": "blah blah blah",
    },
)

docker_build(
    name = "notop_with_double_env",
    base = ":notop_with_env",
    env = {
        "baz": "${foo} $bar",
    },
)

docker_build(
    name = "notop_link_with_files_base",
    base = ":notop_files_base",
    symlinks = {
        "/usr/bin/java": "/bar",
    },
)

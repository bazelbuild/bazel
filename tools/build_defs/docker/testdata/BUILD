package(
    default_visibility = [
        "//tools/build_rules/docker:__subpackages__",
    ],
)

load("/tools/build_rules/docker/docker", "docker_build")

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
    deps = ["//tools/build_rules/docker:archive"],
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

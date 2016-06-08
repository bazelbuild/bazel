package(
    default_visibility = [
        "//tools/build_defs/docker:__subpackages__",
    ],
)

load("//tools/build_defs/docker:docker.bzl", "docker_build")

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//tools/build_defs/docker/testdata/test:srcs",
    ],
)

genrule(
    name = "gen",
    outs = ["gen.out"],
    cmd = "echo generated > $@",
)

docker_build(
    name = "dummy_repository",
    files = ["foo"],
    repository = "gcr.io/dummy",
)

docker_build(
    name = "no_data_path_image",
    files = ["//tools/build_defs/docker/testdata/test:test-data"],
    mode = "0644",
)

docker_build(
    name = "data_path_image",
    data_path = ".",
    files = ["//tools/build_defs/docker/testdata/test:test-data"],
    mode = "0644",
)

docker_build(
    name = "absolute_data_path_image",
    data_path = "/tools/build_defs",
    files = ["//tools/build_defs/docker/testdata/test:test-data"],
    mode = "0644",
)

docker_build(
    name = "root_data_path_image",
    data_path = "/",
    files = ["//tools/build_defs/docker/testdata/test:test-data"],
    mode = "0644",
)

docker_build(
    name = "gen_image",
    files = [":gen"],
    mode = "0644",
)

docker_build(
    name = "files_base",
    files = ["foo"],
    mode = "0644",
)

docker_build(
    name = "files_with_files_base",
    base = ":files_base",
    files = ["bar"],
    mode = "0644",
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
    name = "directory_with_tar_base",
    base = ":tar_base",
    directory = "/foo",
    tars = ["two.tar"],
)

docker_build(
    name = "workdir_with_tar_base",
    base = ":tar_base",
    workdir = "/tmp",
)

docker_build(
    name = "files_with_tar_base",
    base = ":tar_base",
    files = ["bar"],
    mode = "0644",
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
    mode = "0644",
    ports = ["8080"],
    tars = ["two.tar"],
)

# TODO(mattmoor): Test scalar cmd
docker_build(
    name = "derivative_with_shadowed_cmd",
    base = ":base_with_entrypoint",
    cmd = ["shadowed-arg"],
    files = ["foo"],
    mode = "0644",
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
    mode = "0644",
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
    deps = ["//tools/build_defs/pkg:archive"],
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
    name = "with_label",
    base = ":base_with_volume",
    labels = {
        "com.example.foo": "@blah.json",
        "com.example.bar": "@blah.json",
        "com.example.baz": "qux",
    },
)

docker_build(
    name = "with_double_label",
    base = ":with_label",
    labels = {
        "com.example.qux": "@blah-blah.json",
    },
)

[genrule(
    name = "label-" + n,
    outs = ["%s.json" % n],
    cmd = "echo -n '{\"name\": \"%s\"}' > $@" % n,
) for n in [
    "blah",
    "blah-blah",
]]

docker_build(
    name = "with_user",
    base = ":base_with_volume",
    user = "nobody",
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
    deps = ["//tools/build_defs/pkg:archive"],
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
    mode = "0644",
)

docker_build(
    name = "notop_tar_with_tar_base",
    base = ":notop_tar_base",
    tars = ["two.tar"],
)

docker_build(
    name = "notop_directory_with_tar_base",
    base = ":notop_tar_base",
    directory = "/foo",
    tars = ["two.tar"],
)

docker_build(
    name = "notop_files_with_tar_base",
    base = ":notop_tar_base",
    files = ["bar"],
    mode = "0644",
)

docker_build(
    name = "notop_workdir_with_tar_base",
    base = ":notop_tar_base",
    workdir = "/tmp",
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
    mode = "0644",
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
    name = "notop_with_label",
    base = ":notop_base_with_volume",
    labels = {
        "com.example.foo": "@blah.json",
        "com.example.bar": "@blah.json",
        "com.example.baz": "qux",
    },
)

docker_build(
    name = "notop_with_double_label",
    base = ":notop_with_label",
    labels = {
        "com.example.qux": "@blah-blah.json",
    },
)

docker_build(
    name = "notop_with_user",
    base = ":notop_base_with_volume",
    user = "nobody",
)

docker_build(
    name = "notop_link_with_files_base",
    base = ":notop_files_base",
    symlinks = {
        "/usr/bin/java": "/bar",
    },
)

# Generate a dummy debian package with a test/ directory
py_binary(
    name = "gen_deb",
    srcs = ["gen_deb.py"],
)

genrule(
    name = "generate_deb",
    outs = ["gen.deb"],
    cmd = "$(location :gen_deb) $@",
    tools = [":gen_deb"],
)

# Bot gen.deb and test-data has a test directory, it should appear
# only once in the resulting layer.
docker_build(
    name = "extras_with_deb",
    data_path = ".",
    debs = [":gen.deb"],
    tars = ["extras.tar"],
)

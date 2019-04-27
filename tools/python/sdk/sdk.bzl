def get_binary_url(version):
    arch = "Python-{}.tar.xz".format(version)
    return "https://www.python.org/ftp/python/{}/{}".format(version, arch)

def configure(ctx):
    ctx.report_progress("Configure")
    res = ctx.execute([
        "./configure",
        "--prefix=" + str(ctx.path("./"))
    ], working_directory="./src")

    if res.return_code:
        fail("error during configuring Python SDK:\n" + res.stdout + res.stderr)

def build(ctx):
    ctx.report_progress("Build")
    res = ctx.execute(["make"], working_directory="./src")

    if res.return_code:
        fail("error during building Python SDK:\n" + res.stdout + res.stderr)

def install(ctx):
    ctx.report_progress("Install")
    res = ctx.execute([
        "make", "install"
    ], working_directory="./src")

    if res.return_code:
        fail("error during installing Python SDK:\n" + res.stdout + res.stderr)

def _detect_host_platform(ctx):
    host = None

    if ctx.os.name == "linux":
        host = "linux_amd64"
    elif ctx.os.name == "mac os x":
        host = "darwin_amd64"
    elif ctx.os.name.startswith("windows"):
        host = None # TODO: support windows
    elif ctx.os.name == "freebsd":
        host = "freebsd_amd64"

    return host

def _download_sdk_impl(ctx):
    major, minor = [int(v) for v in str(ctx.attr.version).split(".")[:2]]
    if (major != 2 and major != 3 ):
        fail("The version({}) is not supported".format(ctx.attr.version))

    host = _detect_host_platform(ctx)
    if not host:
        fail("Current host OS({}) is not supported".format(ctx.os.name))
    url = get_binary_url(ctx.attr.version)
    prefix = "Python-{}".format(ctx.attr.version)

    ctx.download_and_extract(
        url,
        output = "src",
        stripPrefix = prefix,
        sha256 = ctx.attr.sha256
    )
    configure(ctx)
    build(ctx)
    install(ctx)
    ctx.delete("src")

    ctx.template(
        "BUILD",
        Label("@bazel_tools//tools/python/sdk:BUILD.sdk"),
        executable = False,
        substitutions = {
            "{PY}": "PY3" if major == 3 else "PY2",
            "{name}": "python{}.{}".format(major, minor)
        }
    )

py_download_sdk = repository_rule(
    implementation = _download_sdk_impl,
    attrs = {
        "version": attr.string(
            mandatory=True,
            doc="Python version"
        ),
        "sha256": attr.string()
    },
    doc = """\
load(@bazel_tools//tools/python/sdk:sdk.bzl", "py_download_sdk")

py_download_sdk(
    name = "<workspace_name>",
    version = "<python_sem_version">
)

after that you have available runtime as @<workspace_name>//:runtime
and can include in `py_runtime_pair`

load("@bazel_tools//tools/python:toolchain.bzl", "py_runtime_pair")
py_runtime_pair(
    name = "downloaded_sdk_runtimes",
    py2_runtime = "@<workspace_name>//:runtime", # if @<workspace_name>//:runtime is PY2
    py3_runtime = "@<workspace_name>//:runtime", # if @<workspace_name>//:runtime is PY3
)

and use it in toolchain

toolchain(
    name = "downloaded_sdk_toolchain",
    toolchain = ":downloaded_sdk_runtimes",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type"
)
"""
)

load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

def _impl(ctx):
    ctx.actions.write(
        output = ctx.outputs.output,
        content = """
        set -ex
        $0.runfiles/_main/{generate_modmap} $@ {compiler}
        """.format(
            generate_modmap = ctx.executable._generate_modmap.short_path,
            compiler = ctx.attr.compiler,
        ),
        is_executable = True,
    )
    return [DefaultInfo(
        executable = ctx.outputs.output,
        runfiles = ctx.runfiles(
            files = [ctx.executable._generate_modmap],
            collect_data = True
        ),
    )]

_generate_modmap_wrapper = rule(
    implementation = _impl,
    attrs = {
        "compiler": attr.string(
            mandatory = True,
            doc = "The compiler to use.",
        ),
        "_generate_modmap": attr.label(
            default = "@bazel_tools//tools/cpp:generate-modmap",
            executable = True,
            cfg = "exec",
        ),
        "output": attr.output(
            mandatory = True,
            doc = "The output file."
        ),
    },
    executable = True,
)

def generate_modmap_wrapper(
        name,
        compiler):
    _generate_modmap_wrapper(
        name = "gen_" + name,
        compiler = compiler,
        output = name + ".sh",
    )
    sh_binary(
        name = name,
        srcs = [":gen_" + name],
        visibility = ["//visibility:public"],
    )

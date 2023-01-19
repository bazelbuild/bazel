def _get_runfiles_path(ctx, file):
    if file.short_path.startswith("../"):
        return file.short_path[3:]
    else:
        return ctx.workspace_name + "/" + file.short_path

def _get_wrapper(ctx, linux_path, darwin_path):
    return """#!/bin/bash
if [[ "$OSTYPE" == linux* ]]; then
    exec $0.runfiles/{linux_path} "$@"
elif [[ "$OSTYPE" == darwin* ]]; then
    exec $0.runfiles/{darwin_path} "$@"
else
    echo "Unknown platform $OSTYPE" >&2
    exit 1
fi
""".format(linux_path = linux_path, darwin_path = darwin_path)

def _impl(ctx):
    wrapper = ctx.actions.declare_file(ctx.label.name)

    content = _get_wrapper(
        ctx,
        _get_runfiles_path(ctx, ctx.executable.linux),
        _get_runfiles_path(ctx, ctx.executable.darwin),
    )

    ctx.actions.write(
        output = wrapper,
        content = content,
        is_executable = True,
    )

    runfiles = ctx.runfiles(files = ctx.files.linux + ctx.files.darwin).merge_all([
        ctx.attr.linux[DefaultInfo].default_runfiles,
        ctx.attr.darwin[DefaultInfo].default_runfiles,
    ])

    return DefaultInfo(
        executable = wrapper,
        runfiles = runfiles,
    )

universal_binary = rule(
    implementation = _impl,
    executable = True,
    attrs = {
        "linux": attr.label(
            cfg = "exec",
            mandatory = True,
            executable = True,
            allow_files = True,
        ),
        "darwin": attr.label(
            cfg = "exec",
            mandatory = True,
            executable = True,
            allow_files = True,
        ),
    },
)

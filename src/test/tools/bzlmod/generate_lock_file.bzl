def _generate_lock_file_impl(ctx):
    """Run `bazel mod deps` in an empty workspace to generate a MODULE.bazel.lock file."""
    lock_file = ctx.actions.declare_file("MODULE.bazel.lock")
    script_content = """
    mkdir empty_workspace
    cd empty_workspace
    touch WORKSPACE
    ../{bazel_binary} mod deps
    cd ..
    cp ./empty_workspace/MODULE.bazel.lock {lock_file}
    """.format(bazel_binary=ctx.executable.bazel_binary.path, lock_file=lock_file.path)

    script_file = ctx.actions.declare_file("run_bazel_mod_deps.sh")
    ctx.actions.write(script_file, script_content, is_executable=True)

    # Execute bazel mod deps to generate the lock file
    ctx.actions.run(
        inputs = [ctx.executable.bazel_binary, script_file],
        outputs = [lock_file],
        executable = script_file,
        execution_requirements = {
            "local": "1",  # Ensure this runs locally
        },
    )

    return [DefaultInfo(files = depset([lock_file]))]

generate_lock_file = rule(
    implementation = _generate_lock_file_impl,
    attrs = {
        "bazel_binary": attr.label(
            doc = "Label of the bazel binary",
            default = "//src:bazel",
            cfg = "target", # Avoid re-compiling Bazel for the host platform
            executable = True,
            allow_files = True,
        ),
    },
    outputs = {
        "lock_file": "MODULE.bazel.lock",
    },
    doc = "Generates a MODULE.bazel.lock file by running `bazel mod deps` in an empty workspace.",
)

"""Golden file testing utilities.

This is a classic case of rule macros where public functions wrap private rules.
"""

def _golden_file_test_impl(ctx):
    outputs = [ctx.outputs.executable]

    content = "\n".join([
        """
if diff -u {dst} {src}; then
    echo "files contents are identical"
else
    echo "files differ"
    exit 1
fi
""".format(
            src = src.short_path,
            dst = src.short_path + ".golden",
        )
        for src in ctx.files.srcs
    ])

    ctx.actions.write(
        ctx.outputs.executable,
        "set -x\n\n" + content,
        is_executable = True,
    )
    return [
        DefaultInfo(
            files = depset(outputs),
            runfiles = ctx.runfiles(files = ctx.files.srcs + ctx.files.goldens),
        ),
    ]

_golden_file_test = rule(
    doc = "Asserts the two files are equal",
    implementation = _golden_file_test_impl,
    attrs = {
        "srcs": attr.label_list(
            doc = "the source files under test",
            allow_files = True,
            mandatory = True,
        ),
        "goldens": attr.label_list(
            doc = "the expected golden files",
            allow_files = True,
            mandatory = True,
        ),
    },
    executable = True,
    test = True,
)

def _golden_file_update_impl(ctx):
    outputs = [ctx.outputs.executable]

    content = "\n".join([
        """
cp -f {src} {dst}
echo "Updated golden file: {dst}"
""".format(
            src = src.short_path,
            dst = "$BUILD_WORKING_DIRECTORY/%s/%s.golden" % (ctx.label.package, src.basename),
        )
        for src in ctx.files.srcs
    ])

    ctx.actions.write(
        ctx.outputs.executable,
        "set -x\n\n" + content,
        is_executable = True,
    )
    return [
        DefaultInfo(
            files = depset(outputs),
            runfiles = ctx.runfiles(files = ctx.files.srcs),
        ),
    ]

_golden_file_update = rule(
    doc = "Updates golden files with current source files",
    implementation = _golden_file_update_impl,
    attrs = {
        "srcs": attr.label_list(
            doc = "the source files to update",
            allow_files = True,
            mandatory = True,
        ),
    },
    executable = True,
)

def golden_file_test(name, srcs, **kwargs):
    """Creates both a test and an update target for golden file testing.

    Args:
        name: Name of the test target
        srcs: Source files to compare against golden files
        **kwargs: Additional arguments passed to the underlying rule
    """
    tags = kwargs.get("tags", [])
    _golden_file_test(
        name = name,
        srcs = srcs,
        goldens = native.glob(["*.golden"]),
        tags = tags,
        **kwargs
    )
    _golden_file_update(
        name = name + ".update",
        srcs = srcs,
        tags = tags,
    )

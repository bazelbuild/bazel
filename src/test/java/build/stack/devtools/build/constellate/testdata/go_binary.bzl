"""Private rule definitions for go_binary."""

# Fake provider for testing - must be defined before use
GoLibrary = provider(
    doc = "Go library provider",
    fields = ["importpath"],
)

def _go_binary_impl(ctx):
    """Implementation of the go_binary rule."""
    # Simplified implementation for testing
    output = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(output, "binary content")
    return [DefaultInfo(files = depset([output]))]

def _go_non_executable_binary_impl(ctx):
    """Implementation of the go_non_executable_binary rule."""
    # Simplified implementation for testing
    output = ctx.actions.declare_file(ctx.label.name + ".so")
    ctx.actions.write(output, "shared library content")
    return [DefaultInfo(files = depset([output]))]

go_binary = rule(
    doc = "Builds an executable from Go source code",
    implementation = _go_binary_impl,
    attrs = {
        "srcs": attr.label_list(
            doc = "The list of Go source files to compile",
            allow_files = [".go"],
            mandatory = True,
        ),
        "deps": attr.label_list(
            doc = "List of Go libraries this binary depends on",
            providers = [GoLibrary],
        ),
        "linkmode": attr.string(
            doc = "Determines how the binary is linked",
            default = "normal",
            values = ["normal", "pie", "c-shared", "c-archive"],
        ),
        "goos": attr.string(
            doc = "The target operating system",
        ),
        "goarch": attr.string(
            doc = "The target architecture",
        ),
    },
    executable = True,
)

go_non_executable_binary = rule(
    doc = "Builds a non-executable binary (shared library or archive)",
    implementation = _go_non_executable_binary_impl,
    attrs = {
        "srcs": attr.label_list(
            doc = "The list of Go source files to compile",
            allow_files = [".go"],
            mandatory = True,
        ),
        "deps": attr.label_list(
            doc = "List of Go libraries this binary depends on",
            providers = [GoLibrary],
        ),
        "linkmode": attr.string(
            doc = "Determines how the binary is linked",
            default = "c-shared",
            values = ["c-shared", "c-archive"],
        ),
    },
)

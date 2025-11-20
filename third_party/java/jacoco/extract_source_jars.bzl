def _extract_source_jars_impl(ctx):
    source_jars = []
    for dep in ctx.attr.deps:
        if JavaInfo in dep:
            source_jars.extend(dep[JavaInfo].source_jars)

    if not source_jars:
        fail("No source jars found in dependencies")

    # Create a symbolic link to the first source jar (or copy it)
    # Since we want specific output names, we should probably just return the files
    # But genrule expects specific outputs.
    # Let's try to just return the files in DefaultInfo if we can,
    # but the user wants to use them in a genrule or similar.
    # Actually, if we just want to expose them as a filegroup-like target but with specific names?
    # The user wants to "extract" them.

    # If we use a filegroup, we get the original names.
    # If we want to rename them, we need to copy them.

    outs = []
    for i, jar in enumerate(source_jars):
        # We can't easily rename them dynamically without knowing the count ahead of time if we use a rule that outputs files.
        # But we can use a rule that outputs a DefaultInfo with the files, and then use that in the genrule.
        pass

    return [DefaultInfo(files = depset(source_jars))]

extract_source_jars = rule(
    implementation = _extract_source_jars_impl,
    attrs = {
        "deps": attr.label_list(providers = [JavaInfo]),
    },
)

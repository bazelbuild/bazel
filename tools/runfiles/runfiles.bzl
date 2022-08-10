RunfilesLibraryInfo = provider(
    doc = """
A marker for targets providing runfiles lookup functionality.

Rules may choose to emit additional information required to locate runfiles at runtime if this provider is present on a direct dependency.

Note: At this point, neither Bazel nor native rules check for the presence of this provider.
""",
)

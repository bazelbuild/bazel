"""Test file with correctly formatted docstring."""

def cgo_configure_correct(go, srcs, cdeps, cppopts, copts, cxxopts, clinkopts):
    """cgo_configure returns the inputs and compile / link options
    that are required to build a cgo archive.

    Args:
        go: a GoContext.
        srcs: list of source files being compiled. Include options are added
            for the headers.
        cdeps: list of Targets for C++ dependencies. Include and link options
            may be added.
        cppopts: list of C preprocessor options for the library.
        copts: list of C compiler options for the library.
        cxxopts: list of C++ compiler options for the library.
        clinkopts: list of linker options for the library.

    Returns:
        A struct containing various configuration fields.
    """
    # Dummy implementation
    return struct(
        inputs = depset([]),
        deps = depset([]),
        runfiles = None,
        cppopts = cppopts,
        copts = copts,
        cxxopts = cxxopts,
        objcopts = [],
        objcxxopts = [],
        clinkopts = clinkopts,
    )

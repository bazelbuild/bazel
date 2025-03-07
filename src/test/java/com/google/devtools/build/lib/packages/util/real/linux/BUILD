load("@rules_cc//cc:defs.bzl", "cc_toolchain")
load("@rules_cc//cc/private/toolchain:unix_cc_toolchain_config.bzl", "cc_toolchain_config")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

cc_toolchain(
    name = "cc-compiler-k8-llvm.k8",
    all_files = ":empty",
    applicable_licenses = [],
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = True,
    tags = ["manual"],
    toolchain_config = ":cc-compiler-k8-llvm.k8-config",
    toolchain_identifier = "local",
)

cc_toolchain_config(
    name = "cc-compiler-k8-llvm.k8-config",
    abi_libc_version = "local",
    abi_version = "local",
    applicable_licenses = [],
    compiler = "clang",
    coverage_compile_flags = ["--coverage"],
    coverage_link_flags = ["--coverage"],
    cpu = "k8",
    cxx_builtin_include_directories = [
        # Effectively disable the builtin includes check to make this
        # non-hermetic toolchain definition more portable.
        "/",
    ],
    dbg_compile_flags = ["-g"],
    host_system_name = "local",
    link_flags = [
        "-B/usr/bin",
        "-pass-exit-codes",
    ],
    opt_compile_flags = [
        "-g0",
        "-O2",
    ],
    tags = ["manual"],
    target_libc = "local",
    target_system_name = "local",
    tool_paths = {
        "ar": "/usr/bin/ar",
        "ld": "/usr/bin/ld",
        "cpp": "/usr/bin/cpp",
        "gcc": "/usr/bin/clang",
        "dwp": "/usr/bin/dwp",
        "gcov": "/usr/bin/gcov",
        "nm": "/usr/bin/nm",
        "objcopy": "/usr/bin/objcopy",
        "objdump": "/usr/bin/objdump",
        "strip": "/usr/bin/strip",
    },
    toolchain_identifier = "local",
    unfiltered_compile_flags = [
        "-no-canonical-prefixes",
        "-Wno-builtin-macro-redefined",
        "-D__DATE__=\"redacted\"",
        "-D__TIMESTAMP__=\"redacted\"",
        "-D__TIME__=\"redacted\"",
    ],
)

filegroup(
    name = "empty",
    srcs = [],
    applicable_licenses = [],
)

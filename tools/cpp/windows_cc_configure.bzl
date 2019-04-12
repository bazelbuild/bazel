# pylint: disable=g-bad-file-header
# Copyright 2016 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuring the C++ toolchain on Windows."""

load(
    "@bazel_tools//tools/cpp:lib_cc_configure.bzl",
    "auto_configure_fail",
    "auto_configure_warning",
    "escape_string",
    "execute",
    "get_env_var",
    "get_starlark_list",
    "is_cc_configure_debug",
    "resolve_labels",
)

def _auto_configure_warning_maybe(repository_ctx, msg):
    """Output warning message when CC_CONFIGURE_DEBUG is enabled."""
    if is_cc_configure_debug(repository_ctx):
        auto_configure_warning(msg)

def _get_escaped_windows_msys_starlark_content(repository_ctx, use_mingw = False):
    """Return the content of msys cc toolchain rule."""
    bazel_sh = get_env_var(repository_ctx, "BAZEL_SH", "", False).replace("\\", "/").lower()
    tokens = bazel_sh.rsplit("/", 1)
    msys_root = ""
    if tokens[0].endswith("/usr/bin"):
        msys_root = tokens[0][:len(tokens[0]) - len("usr/bin")]
    elif tokens[0].endswith("/bin"):
        msys_root = tokens[0][:len(tokens[0]) - len("bin")]
    prefix = "mingw64" if use_mingw else "usr"
    tool_path_prefix = escape_string(msys_root) + prefix
    tool_bin_path = tool_path_prefix + "/bin"
    tool_path = {}

    for tool in ["ar", "compat-ld", "cpp", "dwp", "gcc", "gcov", "ld", "nm", "objcopy", "objdump", "strip"]:
        if msys_root:
            tool_path[tool] = tool_bin_path + "/" + tool
        else:
            tool_path[tool] = "msys_gcc_installation_error.bat"
    tool_paths = (
        '        tool_path (name= "ar", path= "%s"),\n' % tool_path["ar"] +
        '        tool_path (name= "compat-ld", path= "%s"),\n' % tool_path["ld"] +
        '        tool_path (name= "cpp", path= "%s"),\n' % tool_path["cpp"] +
        '        tool_path (name= "dwp", path= "%s"),\n' % tool_path["dwp"] +
        '        tool_path (name= "gcc", path= "%s"),\n' % tool_path["gcc"] +
        '        tool_path (name= "gcov", path= "%s"),\n' % tool_path["gcov"] +
        '        tool_path (name= "ld", path= "%s"),\n' % tool_path["ld"] +
        '        tool_path (name= "nm", path= "%s"),\n' % tool_path["nm"] +
        '        tool_path (name= "objcopy", path= "%s"),\n' % tool_path["objcopy"] +
        '        tool_path (name= "objdump", path= "%s"),\n' % tool_path["objdump"] +
        '        tool_path (name= "strip", path= "%s"),\n' % tool_path["strip"]
    )
    include_directories = ('        "%s/",\n        ' % tool_path_prefix) if msys_root else ""
    artifact_name_patterns = '        artifact_name_pattern(category_name="executable", prefix="", extension=".exe"),'
    return tool_paths, tool_bin_path, include_directories, artifact_name_patterns

def _get_system_root(repository_ctx):
    """Get System root path on Windows, default is C:\\\Windows. Doesn't %-escape the result."""
    if "SYSTEMROOT" in repository_ctx.os.environ:
        return escape_string(repository_ctx.os.environ["SYSTEMROOT"])
    _auto_configure_warning_maybe(repository_ctx, "SYSTEMROOT is not set, using default SYSTEMROOT=C:\\Windows")
    return "C:\\Windows"

def _add_system_root(repository_ctx, env):
    """Running VCVARSALL.BAT and VCVARSQUERYREGISTRY.BAT need %SYSTEMROOT%\\\\system32 in PATH."""
    if "PATH" not in env:
        env["PATH"] = ""
    env["PATH"] = env["PATH"] + ";" + _get_system_root(repository_ctx) + "\\system32"
    return env

def find_vc_path(repository_ctx):
    """Find Visual C++ build tools install path. Doesn't %-escape the result."""

    # 1. Check if BAZEL_VC or BAZEL_VS is already set by user.
    if "BAZEL_VC" in repository_ctx.os.environ:
        return repository_ctx.os.environ["BAZEL_VC"]

    if "BAZEL_VS" in repository_ctx.os.environ:
        return repository_ctx.os.environ["BAZEL_VS"] + "\\VC\\"
    _auto_configure_warning_maybe(repository_ctx, "'BAZEL_VC' is not set, " +
                                                  "start looking for the latest Visual C++ installed.")

    # 2. Check if VS%VS_VERSION%COMNTOOLS is set, if true then try to find and use
    # vcvarsqueryregistry.bat / VsDevCmd.bat to detect VC++.
    _auto_configure_warning_maybe(repository_ctx, "Looking for VS%VERSION%COMNTOOLS environment variables, " +
                                                  "eg. VS140COMNTOOLS")
    for vscommontools_env, script in [
        ("VS160COMNTOOLS", "VsDevCmd.bat"),
        ("VS150COMNTOOLS", "VsDevCmd.bat"),
        ("VS140COMNTOOLS", "vcvarsqueryregistry.bat"),
        ("VS120COMNTOOLS", "vcvarsqueryregistry.bat"),
        ("VS110COMNTOOLS", "vcvarsqueryregistry.bat"),
        ("VS100COMNTOOLS", "vcvarsqueryregistry.bat"),
        ("VS90COMNTOOLS", "vcvarsqueryregistry.bat"),
    ]:
        if vscommontools_env not in repository_ctx.os.environ:
            continue
        script = repository_ctx.os.environ[vscommontools_env] + "\\" + script
        if not repository_ctx.path(script).exists:
            continue
        repository_ctx.file(
            "get_vc_dir.bat",
            "@echo off\n" +
            "call \"" + script + "\"\n" +
            "echo %VCINSTALLDIR%",
            True,
        )
        env = _add_system_root(repository_ctx, repository_ctx.os.environ)
        vc_dir = execute(repository_ctx, ["./get_vc_dir.bat"], environment = env)

        _auto_configure_warning_maybe(repository_ctx, "Visual C++ build tools found at %s" % vc_dir)
        return vc_dir

    # 3. User might have purged all environment variables. If so, look for Visual C++ in registry.
    # Works for Visual Studio 2017 and older. (Does not work for Visual Studio 2019 Preview.)
    # TODO(laszlocsomor): check if "16.0" also has this registry key, after VS 2019 is released.
    _auto_configure_warning_maybe(repository_ctx, "Looking for Visual C++ through registry")
    reg_binary = _get_system_root(repository_ctx) + "\\system32\\reg.exe"
    vc_dir = None
    for key, suffix in (("VC7", ""), ("VS7", "\\VC")):
        for version in ["15.0", "14.0", "12.0", "11.0", "10.0", "9.0", "8.0"]:
            if vc_dir:
                break
            result = repository_ctx.execute([reg_binary, "query", "HKEY_LOCAL_MACHINE\\SOFTWARE\\Wow6432Node\\Microsoft\\VisualStudio\\SxS\\" + key, "/v", version])
            _auto_configure_warning_maybe(repository_ctx, "registry query result for VC %s:\n\nSTDOUT(start)\n%s\nSTDOUT(end)\nSTDERR(start):\n%s\nSTDERR(end)\n" %
                                                          (version, result.stdout, result.stderr))
            if not result.stderr:
                for line in result.stdout.split("\n"):
                    line = line.strip()
                    if line.startswith(version) and line.find("REG_SZ") != -1:
                        vc_dir = line[line.find("REG_SZ") + len("REG_SZ"):].strip() + suffix
    if vc_dir:
        _auto_configure_warning_maybe(repository_ctx, "Visual C++ build tools found at %s" % vc_dir)
        return vc_dir

    # 4. Check default directories for VC installation
    _auto_configure_warning_maybe(repository_ctx, "Looking for default Visual C++ installation directory")
    program_files_dir = get_env_var(repository_ctx, "PROGRAMFILES(X86)", default = "C:\\Program Files (x86)", enable_warning = True)
    for path in [
        "Microsoft Visual Studio\\2019\\Preview\\VC",
        "Microsoft Visual Studio\\2019\\BuildTools\\VC",
        "Microsoft Visual Studio\\2019\\Community\\VC",
        "Microsoft Visual Studio\\2019\\Professional\\VC",
        "Microsoft Visual Studio\\2019\\Enterprise\\VC",
        "Microsoft Visual Studio\\2017\\BuildTools\\VC",
        "Microsoft Visual Studio\\2017\\Community\\VC",
        "Microsoft Visual Studio\\2017\\Professional\\VC",
        "Microsoft Visual Studio\\2017\\Enterprise\\VC",
        "Microsoft Visual Studio 14.0\\VC",
    ]:
        path = program_files_dir + "\\" + path
        if repository_ctx.path(path).exists:
            vc_dir = path
            break

    if not vc_dir:
        _auto_configure_warning_maybe(repository_ctx, "Visual C++ build tools not found.")
        return None
    _auto_configure_warning_maybe(repository_ctx, "Visual C++ build tools found at %s" % vc_dir)
    return vc_dir

def _is_vs_2017_or_2019(vc_path):
    """Check if the installed VS version is Visual Studio 2017."""

    # In VS 2017 and 2019, the location of VC is like:
    # C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\
    # In VS 2015 or older version, it is like:
    # C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\
    return vc_path.find("2017") != -1 or vc_path.find("2019") != -1

def _find_vcvarsall_bat_script(repository_ctx, vc_path):
    """Find vcvarsall.bat script. Doesn't %-escape the result."""
    if _is_vs_2017_or_2019(vc_path):
        vcvarsall = vc_path + "\\Auxiliary\\Build\\VCVARSALL.BAT"
    else:
        vcvarsall = vc_path + "\\VCVARSALL.BAT"

    if not repository_ctx.path(vcvarsall).exists:
        return None

    return vcvarsall

def setup_vc_env_vars(repository_ctx, vc_path):
    """Get environment variables set by VCVARSALL.BAT. Doesn't %-escape the result!"""
    vcvarsall = _find_vcvarsall_bat_script(repository_ctx, vc_path)
    if not vcvarsall:
        return None
    repository_ctx.file(
        "get_env.bat",
        "@echo off\n" +
        "call \"" + vcvarsall + "\" amd64 > NUL \n" +
        "echo PATH=%PATH%,INCLUDE=%INCLUDE%,LIB=%LIB%,WINDOWSSDKDIR=%WINDOWSSDKDIR% \n",
        True,
    )
    env = _add_system_root(
        repository_ctx,
        {"PATH": "", "INCLUDE": "", "LIB": "", "WINDOWSSDKDIR": ""},
    )
    envs = execute(repository_ctx, ["./get_env.bat"], environment = env).split(",")
    env_map = {}
    for env in envs:
        key, value = env.split("=", 1)
        env_map[key] = escape_string(value.replace("\\", "\\\\"))
    return env_map

def find_msvc_tool(repository_ctx, vc_path, tool):
    """Find the exact path of a specific build tool in MSVC. Doesn't %-escape the result."""
    tool_path = ""
    if _is_vs_2017_or_2019(vc_path):
        # For VS 2017 and 2019, the tools are under a directory like:
        # C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Tools\MSVC\14.10.24930\bin\HostX64\x64
        dirs = repository_ctx.path(vc_path + "\\Tools\\MSVC").readdir()
        if len(dirs) < 1:
            return None

        # Normally there should be only one child directory under %VC_PATH%\TOOLS\MSVC,
        # but iterate every directory to be more robust.
        for path in dirs:
            tool_path = str(path) + "\\bin\\HostX64\\x64\\" + tool
            if repository_ctx.path(tool_path).exists:
                break
    else:
        # For VS 2015 and older version, the tools are under:
        # C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64
        tool_path = vc_path + "\\bin\\amd64\\" + tool

    if not repository_ctx.path(tool_path).exists:
        return None

    return tool_path.replace("\\", "/")

def _find_missing_vc_tools(repository_ctx, vc_path):
    """Check if any required tool is missing under given VC path."""
    missing_tools = []
    if not _find_vcvarsall_bat_script(repository_ctx, vc_path):
        missing_tools.append("VCVARSALL.BAT")

    for tool in ["cl.exe", "link.exe", "lib.exe", "ml64.exe"]:
        if not find_msvc_tool(repository_ctx, vc_path, tool):
            missing_tools.append(tool)

    return missing_tools

def _is_support_debug_fastlink(repository_ctx, linker):
    """Run linker alone to see if it supports /DEBUG:FASTLINK."""
    if _use_clang_cl(repository_ctx):
        # LLVM's lld-link.exe doesn't support /DEBUG:FASTLINK.
        return False
    result = execute(repository_ctx, [linker], expect_failure = True)
    return result.find("/DEBUG[:{FASTLINK|FULL|NONE}]") != -1

def find_llvm_path(repository_ctx):
    """Find LLVM install path."""

    # 1. Check if BAZEL_LLVM is already set by user.
    if "BAZEL_LLVM" in repository_ctx.os.environ:
        return repository_ctx.os.environ["BAZEL_LLVM"]

    _auto_configure_warning_maybe(repository_ctx, "'BAZEL_LLVM' is not set, " +
                                                  "start looking for LLVM installation on machine.")

    # 2. Look for LLVM installation through registry.
    _auto_configure_warning_maybe(repository_ctx, "Looking for LLVM installation through registry")
    reg_binary = _get_system_root(repository_ctx) + "\\system32\\reg.exe"
    llvm_dir = None
    result = repository_ctx.execute([reg_binary, "query", "HKEY_LOCAL_MACHINE\\SOFTWARE\\WOW6432Node\\LLVM\\LLVM"])
    _auto_configure_warning_maybe(repository_ctx, "registry query result for LLVM:\n\nSTDOUT(start)\n%s\nSTDOUT(end)\nSTDERR(start):\n%s\nSTDERR(end)\n" %
                                                  (result.stdout, result.stderr))
    if not result.stderr:
        for line in result.stdout.split("\n"):
            line = line.strip()
            if line.startswith("(Default)") and line.find("REG_SZ") != -1:
                llvm_dir = line[line.find("REG_SZ") + len("REG_SZ"):].strip()
    if llvm_dir:
        _auto_configure_warning_maybe(repository_ctx, "LLVM installation found at %s" % llvm_dir)
        return llvm_dir

    # 3. Check default directories for LLVM installation
    _auto_configure_warning_maybe(repository_ctx, "Looking for default LLVM installation directory")
    program_files_dir = get_env_var(repository_ctx, "PROGRAMFILES", default = "C:\\Program Files", enable_warning = True)
    path = program_files_dir + "\\LLVM"
    if repository_ctx.path(path).exists:
        llvm_dir = path

    if not llvm_dir:
        _auto_configure_warning_maybe(repository_ctx, "LLVM installation not found.")
        return None
    _auto_configure_warning_maybe(repository_ctx, "LLVM installation found at %s" % llvm_dir)
    return llvm_dir

def find_llvm_tool(repository_ctx, llvm_path, tool):
    """Find the exact path of a specific build tool in LLVM. Doesn't %-escape the result."""
    tool_path = llvm_path + "\\bin\\" + tool

    if not repository_ctx.path(tool_path).exists:
        return None

    return tool_path.replace("\\", "/")

def _use_clang_cl(repository_ctx):
    """Returns True if USE_CLANG_CL is set to 1."""
    return repository_ctx.os.environ.get("USE_CLANG_CL", default = "0") == "1"

def _get_clang_version(repository_ctx, clang_cl):
    result = repository_ctx.execute([clang_cl, "-v"])
    if result.return_code != 0:
        auto_configure_fail("Failed to get clang version by running \"%s -v\"" % clang_cl)

    # Stderr should look like "clang version X.X.X ..."
    return result.stderr.strip().split(" ")[2]

def configure_windows_toolchain(repository_ctx):
    """Configure C++ toolchain on Windows."""
    paths = resolve_labels(repository_ctx, [
        "@bazel_tools//tools/cpp:BUILD.static.windows",
        "@bazel_tools//tools/cpp:cc_toolchain_config.bzl.tpl",
        "@bazel_tools//tools/cpp:vc_installation_error.bat.tpl",
        "@bazel_tools//tools/cpp:msys_gcc_installation_error.bat",
    ])

    repository_ctx.symlink(paths["@bazel_tools//tools/cpp:BUILD.static.windows"], "BUILD")
    repository_ctx.symlink(
        paths["@bazel_tools//tools/cpp:msys_gcc_installation_error.bat"],
        "msys_gcc_installation_error.bat",
    )

    vc_path = find_vc_path(repository_ctx)
    missing_tools = None
    if not vc_path:
        repository_ctx.template(
            "vc_installation_error.bat",
            paths["@bazel_tools//tools/cpp:vc_installation_error.bat.tpl"],
            {"%{vc_error_message}": ""},
        )
    else:
        missing_tools = _find_missing_vc_tools(repository_ctx, vc_path)
        if missing_tools:
            message = "\r\n".join([
                "echo. 1>&2",
                "echo Visual C++ build tools seems to be installed at %s 1>&2" % vc_path,
                "echo But Bazel can't find the following tools: 1>&2",
                "echo     %s 1>&2" % ", ".join(missing_tools),
                "echo. 1>&2",
            ])
            repository_ctx.template(
                "vc_installation_error.bat",
                paths["@bazel_tools//tools/cpp:vc_installation_error.bat.tpl"],
                {"%{vc_error_message}": message},
            )

    tool_paths_mingw, tool_bin_path_mingw, inc_dir_mingw, _ = _get_escaped_windows_msys_starlark_content(repository_ctx, use_mingw = True)
    tool_paths, tool_bin_path, inc_dir_msys, artifact_patterns = _get_escaped_windows_msys_starlark_content(repository_ctx)
    if not vc_path or missing_tools:
        repository_ctx.template(
            "cc_toolchain_config.bzl",
            paths["@bazel_tools//tools/cpp:cc_toolchain_config.bzl.tpl"],
            {
                "%{toolchain_identifier}": "msys_x64",
                "%{msvc_env_tmp}": "msvc_not_found",
                "%{msvc_env_path}": "msvc_not_found",
                "%{msvc_env_include}": "msvc_not_found",
                "%{msvc_env_lib}": "msvc_not_found",
                "%{msvc_cl_path}": "vc_installation_error.bat",
                "%{msvc_ml_path}": "vc_installation_error.bat",
                "%{msvc_link_path}": "vc_installation_error.bat",
                "%{msvc_lib_path}": "vc_installation_error.bat",
                "%{msvc_cxx_builtin_include_directories}": "",
                "%{msys_x64_mingw_cxx_content}": get_starlark_list(["-std=gnu++0x"]),
                "%{msys_x64_mingw_link_content}": get_starlark_list(["-lstdc++"]),
                "%{dbg_mode_debug}": "/DEBUG",
                "%{fastbuild_mode_debug}": "/DEBUG",
                "%{compile_content}": "",
                "%{cxx_content}": get_starlark_list(["-std=gnu++0x"]),
                "%{link_content}": get_starlark_list(["-lstdc++"]),
                "%{opt_compile_content}": "",
                "%{opt_link_content}": "",
                "%{unfiltered_content}": "",
                "%{dbg_compile_content}": "",
                "%{cxx_builtin_include_directories}": inc_dir_msys,
                "%{mingw_cxx_builtin_include_directories}": inc_dir_mingw,
                "%{coverage_feature}": "",
                "%{use_coverage_feature}": "",
                "%{supports_start_end_lib}": "",
                "%{use_windows_features}": "windows_features + ",
                "%{abi_version}": "local",
                "%{abi_libc_version}": "local",
                "%{builtin_sysroot}": "",
                "%{compiler}": "msys-gcc",
                "%{host_system_name}": "local",
                "%{target_libc}": "msys",
                "%{target_cpu}": "x64_windows",
                "%{target_system_name}": "local",
                "%{tool_paths}": tool_paths,
                "%{mingw_tool_paths}": tool_paths_mingw,
                "%{artifact_name_patterns}": artifact_patterns,
                "%{tool_bin_path}": tool_bin_path,
                "%{mingw_tool_bin_path}": tool_bin_path_mingw,
            },
        )
        return

    env = setup_vc_env_vars(repository_ctx, vc_path)
    escaped_paths = escape_string(env["PATH"])
    escaped_include_paths = escape_string(env["INCLUDE"])
    escaped_lib_paths = escape_string(env["LIB"])
    escaped_tmp_dir = escape_string(
        get_env_var(repository_ctx, "TMP", "C:\\Windows\\Temp").replace("\\", "\\\\"),
    )

    llvm_path = ""
    if _use_clang_cl(repository_ctx):
        llvm_path = find_llvm_path(repository_ctx)
        if not llvm_path:
            auto_configure_fail("\nUSE_CLANG_CL is set to 1, but Bazel cannot find Clang installation on your system.\n" +
                                "Please install Clang via http://releases.llvm.org/download.html\n")
        cl_path = find_llvm_tool(repository_ctx, llvm_path, "clang-cl.exe")
        link_path = find_llvm_tool(repository_ctx, llvm_path, "lld-link.exe")
        if not link_path:
            link_path = find_msvc_tool(repository_ctx, vc_path, "link.exe")
        lib_path = find_llvm_tool(repository_ctx, llvm_path, "llvm-lib.exe")
        if not lib_path:
            lib_path = find_msvc_tool(repository_ctx, vc_path, "lib.exe")
    else:
        cl_path = find_msvc_tool(repository_ctx, vc_path, "cl.exe")
        link_path = find_msvc_tool(repository_ctx, vc_path, "link.exe")
        lib_path = find_msvc_tool(repository_ctx, vc_path, "lib.exe")

    msvc_ml_path = find_msvc_tool(repository_ctx, vc_path, "ml64.exe")
    escaped_cxx_include_directories = []

    for path in escaped_include_paths.split(";"):
        if path:
            escaped_cxx_include_directories.append("\"%s\"" % path)
    if llvm_path:
        clang_version = _get_clang_version(repository_ctx, cl_path)
        clang_dir = llvm_path + "\\lib\\clang\\" + clang_version
        clang_include_path = (clang_dir + "\\include").replace("\\", "\\\\")
        escaped_cxx_include_directories.append("\"%s\"" % clang_include_path)
        clang_lib_path = (clang_dir + "\\lib\\windows").replace("\\", "\\\\")
        escaped_lib_paths = escaped_lib_paths + ";" + clang_lib_path

    support_debug_fastlink = _is_support_debug_fastlink(repository_ctx, link_path)

    repository_ctx.template(
        "cc_toolchain_config.bzl",
        paths["@bazel_tools//tools/cpp:cc_toolchain_config.bzl.tpl"],
        {
            "%{toolchain_identifier}": "msys_x64",
            "%{msvc_env_tmp}": escaped_tmp_dir,
            "%{msvc_env_path}": escaped_paths,
            "%{msvc_env_include}": escaped_include_paths,
            "%{msvc_env_lib}": escaped_lib_paths,
            "%{msvc_cl_path}": cl_path,
            "%{msvc_ml_path}": msvc_ml_path,
            "%{msvc_link_path}": link_path,
            "%{msvc_lib_path}": lib_path,
            "%{dbg_mode_debug}": "/DEBUG:FULL" if support_debug_fastlink else "/DEBUG",
            "%{fastbuild_mode_debug}": "/DEBUG:FASTLINK" if support_debug_fastlink else "/DEBUG",
            "%{msys_x64_mingw_cxx_content}": get_starlark_list(["-std=gnu++0x"]),
            "%{msys_x64_mingw_link_content}": get_starlark_list(["-lstdc++"]),
            "%{compile_content}": "",
            "%{cxx_content}": get_starlark_list(["-std=gnu++0x"]),
            "%{link_content}": get_starlark_list(["-lstdc++"]),
            "%{opt_compile_content}": "",
            "%{opt_link_content}": "",
            "%{unfiltered_content}": "",
            "%{dbg_compile_content}": "",
            "%{cxx_builtin_include_directories}": inc_dir_msys + ",\n        ".join(escaped_cxx_include_directories),
            "%{msvc_cxx_builtin_include_directories}": "        " + ",\n        ".join(escaped_cxx_include_directories),
            "%{mingw_cxx_builtin_include_directories}": inc_dir_mingw + ",\n        ".join(escaped_cxx_include_directories),
            "%{coverage_feature}": "",
            "%{use_coverage_feature}": "",
            "%{supports_start_end_lib}": "",
            "%{use_windows_features}": "windows_features + ",
            "%{abi_version}": "local",
            "%{abi_libc_version}": "local",
            "%{builtin_sysroot}": "",
            "%{compiler}": "msys-gcc",
            "%{host_system_name}": "local",
            "%{target_libc}": "msys",
            "%{target_cpu}": "x64_windows",
            "%{target_system_name}": "local",
            "%{tool_paths}": tool_paths,
            "%{mingw_tool_paths}": tool_paths_mingw,
            "%{artifact_name_patterns}": artifact_patterns,
            "%{tool_bin_path}": tool_bin_path,
            "%{mingw_tool_bin_path}": tool_bin_path_mingw,
        },
    )

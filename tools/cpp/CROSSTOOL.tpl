major_version: "local"
minor_version: ""
default_target_cpu: "same_as_host"

default_toolchain {
  cpu: "%{cpu}"
  toolchain_identifier: "local"
}

default_toolchain {
  cpu: "armeabi-v7a"
  toolchain_identifier: "stub_armeabi-v7a"
}

default_toolchain {
  cpu: "x64_windows_msvc"
  toolchain_identifier: "vc_14_0_x64"
}

# Android tooling requires a default toolchain for the armeabi-v7a cpu.
toolchain {
  abi_version: "armeabi-v7a"
  abi_libc_version: "armeabi-v7a"
  builtin_sysroot: ""
  compiler: "compiler"
  host_system_name: "armeabi-v7a"
  needsPic: true
  supports_gold_linker: false
  supports_incremental_linker: false
  supports_fission: false
  supports_interface_shared_objects: false
  supports_normalizing_ar: false
  supports_start_end_lib: false
  supports_thin_archives: false
  target_libc: "armeabi-v7a"
  target_cpu: "armeabi-v7a"
  target_system_name: "armeabi-v7a"
  toolchain_identifier: "stub_armeabi-v7a"

  tool_path { name: "ar" path: "/bin/false" }
  tool_path { name: "compat-ld" path: "/bin/false" }
  tool_path { name: "cpp" path: "/bin/false" }
  tool_path { name: "dwp" path: "/bin/false" }
  tool_path { name: "gcc" path: "/bin/false" }
  tool_path { name: "gcov" path: "/bin/false" }
  tool_path { name: "ld" path: "/bin/false" }

  tool_path { name: "nm" path: "/bin/false" }
  tool_path { name: "objcopy" path: "/bin/false" }
  tool_path { name: "objdump" path: "/bin/false" }
  tool_path { name: "strip" path: "/bin/false" }
  linking_mode_flags { mode: DYNAMIC }
}

toolchain {
  toolchain_identifier: "local"
%{content}

  compilation_mode_flags {
    mode: DBG
%{dbg_content}
  }
  compilation_mode_flags {
    mode: OPT
%{opt_content}
  }
  linking_mode_flags { mode: DYNAMIC }
}

toolchain {
  toolchain_identifier: "vc_14_0_x64"
  host_system_name: "local"
  target_system_name: "local"

  abi_version: "local"
  abi_libc_version: "local"
  target_cpu: "x64_windows_msvc"
  compiler: "cl"
  target_libc: "msvcrt140"
  default_python_version: "python2.7"

%{cxx_builtin_include_directory}

  tool_path {
    name: "ar"
    path: "wrapper/bin/msvc_link.bat"
  }
  tool_path {
    name: "cpp"
    path: "wrapper/bin/msvc_cl.bat"
  }
  tool_path {
    name: "gcc"
    path: "wrapper/bin/msvc_cl.bat"
  }
  tool_path {
    name: "gcov"
    path: "wrapper/bin/msvc_nop.bat"
  }
  tool_path {
    name: "ld"
    path: "wrapper/bin/msvc_link.bat"
  }
  tool_path {
    name: "nm"
    path: "wrapper/bin/msvc_nop.bat"
  }
  tool_path {
    name: "objcopy"
    path: "wrapper/bin/msvc_nop.bat"
  }
  tool_path {
    name: "objdump"
    path: "wrapper/bin/msvc_nop.bat"
  }
  tool_path {
    name: "strip"
    path: "wrapper/bin/msvc_nop.bat"
  }
  supports_gold_linker: false
  supports_thin_archives: false
  supports_start_end_lib: false
  supports_interface_shared_objects: false
  supports_incremental_linker: false
  supports_normalizing_ar: true
  needsPic: false

  compiler_flag: "-m64"
  compiler_flag: "/D__inline__=__inline"
  # TODO(pcloudy): Review those flags below, they should be defined by cl.exe
  compiler_flag: "/DOS_WINDOWS=OS_WINDOWS"
  compiler_flag: "/DCOMPILER_MSVC"

  # Don't pollute with GDI macros in windows.h.
  compiler_flag: "/DNOGDI"
  # Don't define min/max macros in windows.h.
  compiler_flag: "/DNOMINMAX"
  compiler_flag: "/DPRAGMA_SUPPORTED"
  # Platform defines.
  compiler_flag: "/D_WIN32_WINNT=0x0600"
  # Turn off warning messages.
  compiler_flag: "/D_CRT_SECURE_NO_DEPRECATE"
  compiler_flag: "/D_CRT_SECURE_NO_WARNINGS"
  compiler_flag: "/D_SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS"
  # Use math constants (M_PI, etc.) from the math library
  compiler_flag: "/D_USE_MATH_DEFINES"

  # Useful options to have on for compilation.
  # Suppress startup banner.
  compiler_flag: "/nologo"
  # Increase the capacity of object files to 2^32 sections.
  compiler_flag: "/bigobj"
  # Allocate 500MB for precomputed headers.
  compiler_flag: "/Zm500"
  # Use unsigned char by default.
  compiler_flag: "/J"
  # Use function level linking.
  compiler_flag: "/Gy"
  # Use string pooling.
  compiler_flag: "/GF"
  # Warning level 3 (could possibly go to 4 in the future).
  compiler_flag: "/W3"
  # Catch both asynchronous (structured) and synchronous (C++) exceptions.
  compiler_flag: "/EHsc"

  # Globally disabled warnings.
  # Don't warn about elements of array being be default initialized.
  compiler_flag: "/wd4351"
  # Don't warn about no matching delete found.
  compiler_flag: "/wd4291"
  # Don't warn about diamond inheritance patterns.
  compiler_flag: "/wd4250"
  # Don't warn about insecure functions (e.g. non _s functions).
  compiler_flag: "/wd4996"

  linker_flag: "-m64"

  feature {
    name: 'include_paths'
    flag_set {
      action: 'preprocess-assemble'
      action: 'c-compile'
      action: 'c++-compile'
      action: 'c++-header-parsing'
      action: 'c++-header-preprocessing'
      action: 'c++-module-compile'
      flag_group {
        flag: '/I%{quote_include_paths}'
      }
      flag_group {
        flag: '/I%{include_paths}'
      }
      flag_group {
        flag: '/I%{system_include_paths}'
      }
    }
  }

  feature {
    name: 'dependency_file'
    flag_set {
      action: 'assemble'
      action: 'preprocess-assemble'
      action: 'c-compile'
      action: 'c++-compile'
      action: 'c++-module-compile'
      action: 'c++-header-preprocessing'
      action: 'c++-header-parsing'
      expand_if_all_available: 'dependency_file'
      flag_group {
        flag: '/DEPENDENCY_FILE'
        flag: '%{dependency_file}'
      }
    }
  }

  # Stop passing -frandom-seed option
  feature {
    name: 'random_seed'
  }

  # This feature is just for enabling flag_set in action_config for -c and -o options during the transitional period
  feature {
    name: 'compile_action_flags_in_flag_set'
  }

  action_config {
    config_name: 'c-compile'
    action_name: 'c-compile'
    tool {
      tool_path: 'wrapper/bin/msvc_cl.bat'
    }
    flag_set {
      flag_group {
        flag: '/c'
        flag: '%{source_file}'
      }
    }
    flag_set {
      expand_if_all_available: 'output_object_file'
      flag_group {
        flag: '/Fo%{output_object_file}'
      }
    }
    flag_set {
      expand_if_all_available: 'output_assembly_file'
      flag_group {
        flag: '/Fa%{output_assembly_file}'
      }
    }
    flag_set {
      expand_if_all_available: 'output_preprocess_file'
      flag_group {
        flag: '/P'
        flag: '/Fi%{output_preprocess_file}'
      }
    }
  }

  action_config {
    config_name: 'c++-compile'
    action_name: 'c++-compile'
    tool {
      tool_path: 'wrapper/bin/msvc_cl.bat'
    }
    flag_set {
      flag_group {
        flag: '/c'
        flag: '%{source_file}'
      }
    }
    flag_set {
      expand_if_all_available: 'output_object_file'
      flag_group {
        flag: '/Fo%{output_object_file}'
      }
    }
    flag_set {
      expand_if_all_available: 'output_assembly_file'
      flag_group {
        flag: '/Fa%{output_assembly_file}'
      }
    }
    flag_set {
      expand_if_all_available: 'output_preprocess_file'
      flag_group {
        flag: '/P'
        flag: '/Fi%{output_preprocess_file}'
      }
    }
  }

  compilation_mode_flags {
    mode: DBG
    compiler_flag: "/DDEBUG=1"
    # This will signal the wrapper that we are doing a debug build, which sets
    # some internal state of the toolchain wrapper. It is intentionally a "-"
    # flag to make this very obvious.
    compiler_flag: "-g"
    compiler_flag: "/Od"
    compiler_flag: "-Xcompilation-mode=dbg"
  }

  compilation_mode_flags {
    mode: FASTBUILD
    compiler_flag: "/DNDEBUG"
    compiler_flag: "/Od"
    compiler_flag: "-Xcompilation-mode=fastbuild"
  }

  compilation_mode_flags {
    mode: OPT
    compiler_flag: "/DNDEBUG"
    compiler_flag: "/O2"
    compiler_flag: "-Xcompilation-mode=opt"
  }
}
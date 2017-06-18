major_version: "local"
minor_version: ""
default_target_cpu: "same_as_host"

default_toolchain {
  cpu: "%{cpu}"
  toolchain_identifier: "%{default_toolchain_name}"
}

default_toolchain {
  cpu: "armeabi-v7a"
  toolchain_identifier: "stub_armeabi-v7a"
}

default_toolchain {
  cpu: "x64_windows_msvc"
  toolchain_identifier: "msvc_x64"
}

default_toolchain {
  cpu: "x64_windows_msys"
  toolchain_identifier: "msys_x64"
}

default_toolchain {
  cpu: "s390x"
  toolchain_identifier: "%{toolchain_name}"
}

default_toolchain {
  cpu: "ios_x86_64"
  toolchain_identifier: "ios_x86_64"
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
  toolchain_identifier: "ios_x86_64"
  host_system_name: "x86_64-apple-macosx"
  target_system_name: "x86_64-apple-ios"
  target_cpu: "ios_x86_64"
  target_libc: "ios"
  compiler: "compiler"
  abi_version: "local"
  abi_libc_version: "local"
  supports_gold_linker: false
  supports_incremental_linker: false
  supports_fission: false
  supports_interface_shared_objects: false
  supports_normalizing_ar: false
  supports_start_end_lib: false

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
  toolchain_identifier: "%{toolchain_name}"
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

%{coverage}
}

toolchain {
  toolchain_identifier: "msvc_x64"
  host_system_name: "local"
  target_system_name: "local"

  abi_version: "local"
  abi_libc_version: "local"
  target_cpu: "x64_windows"
  compiler: "cl"
  target_libc: "msvcrt140"
  default_python_version: "python2.7"

%{cxx_builtin_include_directory}

  tool_path {
    name: "ar"
    path: "%{msvc_lib_path}"
  }
  tool_path {
    name: "cpp"
    path: "%{msvc_cl_path}"
  }
  tool_path {
    name: "gcc"
    path: "%{msvc_cl_path}"
  }
  tool_path {
    name: "gcov"
    path: "wrapper/bin/msvc_nop.bat"
  }
  tool_path {
    name: "ld"
    path: "%{msvc_link_path}"
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
  supports_start_end_lib: false
  supports_interface_shared_objects: false
  supports_incremental_linker: false
  supports_normalizing_ar: true
  needsPic: false

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

  linker_flag: "/MACHINE:X64"

  linker_flag: "/SUBSYSTEM:CONSOLE"

  # Suppress startup banner.
  feature {
    name: "nologo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "c++-header-parsing"
      action: "c++-header-preprocessing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-static-library"
      action: "c++-link-alwayslink-static-library"
      action: "c++-link-pic-static-library"
      action: "c++-link-alwayslink-pic-static-library"
      flag_group {
        flag: "/nologo"
      }
    }
  }

  feature {
    name: "msvc_env"
    env_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "c++-header-parsing"
      action: "c++-header-preprocessing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-static-library"
      action: "c++-link-alwayslink-static-library"
      action: "c++-link-pic-static-library"
      action: "c++-link-alwayslink-pic-static-library"
      env_entry {
        key: "PATH"
        value: "%{msvc_env_path}"
      }
      env_entry {
        key: "INCLUDE"
        value: "%{msvc_env_include}"
      }
      env_entry {
        key: "LIB"
        value: "%{msvc_env_lib}"
      }
      env_entry {
        key: "TMP"
        value: "%{msvc_env_tmp}"
      }
      env_entry {
        key: "TEMP"
        value: "%{msvc_env_tmp}"
      }
    }
  }

  feature {
    name: "use_linker"
    env_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      env_entry {
        key: "USE_LINKER"
        value: "1"
      }
    }
  }

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

  # Stop adding any flag for dotD file, Bazel knows how to parse the output of /showIncludes option
  # TODO(bazel-team): Remove this empty feature. https://github.com/bazelbuild/bazel/issues/2868
  feature {
    name: 'dependency_file'
  }

  # Tell Bazel to parse the output of /showIncludes
  feature {
    name: 'parse_showincludes'
    flag_set {
      action: 'assemble'
      action: 'preprocess-assemble'
      action: 'c-compile'
      action: 'c++-compile'
      action: 'c++-module-compile'
      action: 'c++-header-preprocessing'
      action: 'c++-header-parsing'
      flag_group {
        flag: "/showIncludes"
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
      tool_path: '%{msvc_cl_path}'
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
    implies: 'nologo'
    implies: 'msvc_env'
    implies: 'parse_showincludes'
  }

  action_config {
    config_name: 'c++-compile'
    action_name: 'c++-compile'
    tool {
      tool_path: '%{msvc_cl_path}'
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
    implies: 'nologo'
    implies: 'msvc_env'
    implies: 'parse_showincludes'
  }

  action_config {
    config_name: 'c++-link-executable'
    action_name: 'c++-link-executable'
    tool {
      tool_path: '%{msvc_link_path}'
    }
    implies: 'nologo'
    implies: 'linkstamps'
    implies: 'output_execpath_flags'
    implies: 'input_param_flags'
    implies: 'legacy_link_flags'
    implies: 'linker_param_file'
    implies: 'msvc_env'
    implies: 'use_linker'
  }

  action_config {
    config_name: 'c++-link-dynamic-library'
    action_name: 'c++-link-dynamic-library'
    tool {
      tool_path: '%{msvc_link_path}'
    }
    implies: 'nologo'
    implies: 'shared_flag'
    implies: 'linkstamps'
    implies: 'output_execpath_flags'
    implies: 'input_param_flags'
    implies: 'has_configured_linker_path'
    implies: 'legacy_link_flags'
    implies: 'linker_param_file'
    implies: 'msvc_env'
    implies: 'use_linker'
  }

  action_config {
    config_name: 'c++-link-static-library'
    action_name: 'c++-link-static-library'
    tool {
      tool_path: '%{msvc_lib_path}'
    }
    implies: 'nologo'
    implies: 'archiver_flags'
    implies: 'input_param_flags'
    implies: 'linker_param_file'
    implies: 'msvc_env'
  }

  action_config {
    config_name: 'c++-link-alwayslink-static-library'
    action_name: 'c++-link-alwayslink-static-library'
    tool {
      tool_path: '%{msvc_lib_path}'
    }
    implies: 'nologo'
    implies: 'archiver_flags'
    implies: 'input_param_flags'
    implies: 'linker_param_file'
    implies: 'msvc_env'
  }

  # TODO(pcloudy): The following action_config is listed in MANDATORY_LINK_TARGET_TYPES.
  # But do we really need them on Windows?
  action_config {
    config_name: 'c++-link-pic-static-library'
    action_name: 'c++-link-pic-static-library'
    tool {
      tool_path: '%{msvc_lib_path}'
    }
    implies: 'nologo'
    implies: 'archiver_flags'
    implies: 'input_param_flags'
    implies: 'linker_param_file'
    implies: 'msvc_env'
  }

  action_config {
    config_name: 'c++-link-alwayslink-pic-static-library'
    action_name: 'c++-link-alwayslink-pic-static-library'
    tool {
      tool_path: '%{msvc_lib_path}'
    }
    implies: 'nologo'
    implies: 'archiver_flags'
    implies: 'input_param_flags'
    implies: 'linker_param_file'
    implies: 'msvc_env'
  }

  action_config {
    config_name: 'c++-link-interface-dynamic-library'
    action_name: 'c++-link-interface-dynamic-library'
    tool {
      tool_path: '%{msvc_lib_path}'
    }
    implies: 'nologo'
    implies: 'linker_param_file'
    implies: 'msvc_env'
  }

  feature {
    name: 'generate_pdb_file'
    requires: {
      feature: 'dbg'
    }
    requires: {
      feature: 'fastbuild'
    }
  }

  feature {
    name: 'has_configured_linker_path'
  }

  feature {
    name: 'shared_flag'
    flag_set {
      action: 'c++-link-dynamic-library'
      flag_group {
        flag: '/DLL'
      }
    }
  }

  feature {
    name: 'linkstamps'
    flag_set {
      action: 'c++-link-executable'
      action: 'c++-link-dynamic-library'
      expand_if_all_available: 'linkstamp_paths'
      flag_group {
        flag: '%{linkstamp_paths}'
      }
    }
  }

  feature {
    name: 'output_execpath_flags'
    flag_set {
      expand_if_all_available: 'output_execpath'
      action: 'c++-link-executable'
      action: 'c++-link-dynamic-library'
      flag_group {
        flag: '/OUT:%{output_execpath}'
      }
    }
  }

  feature {
    name: 'archiver_flags'
    flag_set {
      expand_if_all_available: 'output_execpath'
      action: 'c++-link-static-library'
      action: 'c++-link-alwayslink-static-library'
      action: 'c++-link-pic-static-library'
      action: 'c++-link-alwayslink-pic-static-library'
      flag_group {
        flag: '/OUT:%{output_execpath}'
      }
    }
  }

  feature {
    name: 'input_param_flags'
    flag_set {
      expand_if_all_available: 'library_search_directories'
      action: 'c++-link-executable'
      action: 'c++-link-dynamic-library'
      flag_group {
        iterate_over: 'library_search_directories'
        flag: "-L%{library_search_directories}"
      }
    }
    flag_set {
      expand_if_all_available: 'libopts'
      action: 'c++-link-executable'
      action: 'c++-link-dynamic-library'
      flag_group {
        flag: '%{libopts}'
      }
    }
    flag_set {
      expand_if_all_available: 'libraries_to_link'
      action: 'c++-link-executable'
      action: 'c++-link-dynamic-library'
      action: 'c++-link-static-library'
      action: 'c++-link-alwayslink-static-library'
      action: 'c++-link-pic-static-library'
      action: 'c++-link-alwayslink-pic-static-library'
      flag_group {
        iterate_over: 'libraries_to_link'
        flag_group {
          expand_if_equal: {
            variable: 'libraries_to_link.type'
            value: 'object_file_group'
          }
          iterate_over: 'libraries_to_link.object_files'
          flag_group {
            flag: '%{libraries_to_link.object_files}'
          }
        }
        flag_group {
          expand_if_equal: {
            variable: 'libraries_to_link.type'
            value: 'object_file'
          }
          flag_group {
            flag: '%{libraries_to_link.name}'
          }
        }
        flag_group {
          expand_if_equal: {
            variable: 'libraries_to_link.type'
            value: 'interface_library'
          }
          flag_group {
            expand_if_false: 'libraries_to_link.is_whole_archive'
            flag: '%{libraries_to_link.name}'
          }
          flag_group {
            expand_if_true: 'libraries_to_link.is_whole_archive'
            flag: '/WHOLEARCHIVE:%{libraries_to_link.name}'
          }
        }
        flag_group {
          expand_if_equal: {
            variable: 'libraries_to_link.type'
            value: 'static_library'
          }
          flag_group {
            expand_if_false: 'libraries_to_link.is_whole_archive'
            flag: '%{libraries_to_link.name}'
          }
          flag_group {
            expand_if_true: 'libraries_to_link.is_whole_archive'
            flag: '/WHOLEARCHIVE:%{libraries_to_link.name}'
          }
        }
        flag_group {
          expand_if_equal: {
            variable: 'libraries_to_link.type'
            value: 'dynamic_library'
          }
          flag_group {
            expand_if_false: 'libraries_to_link.is_whole_archive'
            flag: '%{libraries_to_link.name}'
          }
          flag_group {
            expand_if_true: 'libraries_to_link.is_whole_archive'
            flag: '/WHOLEARCHIVE:%{libraries_to_link.name}'
          }
        }
        flag_group {
          expand_if_equal: {
            variable: 'libraries_to_link.type'
            value: 'versioned_dynamic_library'
          }
          flag_group {
            expand_if_false: 'libraries_to_link.is_whole_archive'
            flag: '%{libraries_to_link.name}'
          }
          flag_group {
            expand_if_true: 'libraries_to_link.is_whole_archive'
            flag: '/WHOLEARCHIVE:%{libraries_to_link.name}'
          }
        }
      }
    }
  }

  feature {
    name: 'legacy_link_flags'
    flag_set {
      expand_if_all_available: 'legacy_link_flags'
      action: 'c++-link-executable'
      action: 'c++-link-dynamic-library'
      flag_group {
        flag: '%{legacy_link_flags}'
      }
    }
  }

  feature {
    name: 'linker_param_file'
    flag_set {
      expand_if_all_available: 'linker_param_file'
      action: 'c++-link-executable'
      action: 'c++-link-dynamic-library'
      action: 'c++-link-static-library'
      action: 'c++-link-alwayslink-static-library'
      action: 'c++-link-pic-static-library'
      action: 'c++-link-alwayslink-pic-static-library'
      flag_group {
        flag: '@%{linker_param_file}'
      }
    }
  }

  feature {
    name: 'link_crt_library'
    flag_set {
      action: 'c-compile'
      action: 'c++-compile'
      flag_group {
        # The flag is filled by cc_configure.
        # The default option is /MT, set USE_DYNAMIC_CRT=1 to change it to /MD
        flag: "%{crt_option}"
      }
    }
    flag_set {
      action: 'c++-link-executable'
      action: 'c++-link-dynamic-library'
      flag_group {
      # The flag is filled by cc_configure.
        # The default value is libcmt.lib, set USE_DYNAMIC_CRT=1 to change it to msvcrt.lib
        flag: "/DEFAULTLIB:%{crt_library}"
      }
    }
  }

  feature {
    name: 'link_crt_debug_library'
    flag_set {
      action: 'c-compile'
      action: 'c++-compile'
      flag_group {
        # The flag is filled by cc_configure.
        # The default option is /MTd, set USE_DYNAMIC_CRT=1 to change it to /MDd
        flag: "%{crt_debug_option}"
      }
    }
    flag_set {
      action: 'c++-link-executable'
      action: 'c++-link-dynamic-library'
      flag_group {
        # The flag is filled by cc_configure.
        # The default value is libcmtd.lib, set USE_DYNAMIC_CRT=1 to change it to msvcrtd.lib
        flag: "/DEFAULTLIB:%{crt_debug_library}"
      }
    }
  }

  feature {
    name: 'dbg'
    flag_set {
      action: 'c-compile'
      action: 'c++-compile'
      flag_group {
        flag: "/Od"
        flag: "/Z7"
        # This will signal the wrapper that we are doing a debug build, which sets
        # some internal state of the toolchain wrapper. It is intentionally a "-"
        # flag to make this very obvious.
        flag: "-g"
      }
    }
    flag_set {
      action: 'c++-link-executable'
      action: 'c++-link-dynamic-library'
      flag_group {
        flag: "/DEBUG:FULL"
        flag: "/INCREMENTAL:NO"
      }
    }
    implies: 'link_crt_debug_library'
    implies: 'generate_pdb_file'
  }

  feature {
    name: 'fastbuild'
    flag_set {
      action: 'c-compile'
      action: 'c++-compile'
      flag_group {
        flag: "/Od"
        flag: "/Z7"
      }
    }
    flag_set {
      action: 'c++-link-executable'
      action: 'c++-link-dynamic-library'
      flag_group {
        flag: "/DEBUG:FASTLINK"
        flag: "/INCREMENTAL:NO"
      }
    }
    implies: 'link_crt_library'
    implies: 'generate_pdb_file'
  }

  feature {
    name: 'opt'
    flag_set {
      action: 'c-compile'
      action: 'c++-compile'
      flag_group {
        flag: "/O2"
      }
    }
    implies: 'link_crt_library'
  }

%{compilation_mode_content}

}

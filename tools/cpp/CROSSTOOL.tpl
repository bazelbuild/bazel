major_version: "local"
minor_version: ""
toolchain {
  toolchain_identifier: "stub_armeabi-v7a"
  host_system_name: "armeabi-v7a"
  target_system_name: "armeabi-v7a"
  target_cpu: "armeabi-v7a"
  target_libc: "armeabi-v7a"
  compiler: "compiler"
  abi_version: "armeabi-v7a"
  abi_libc_version: "armeabi-v7a"
  tool_path {
    name: "ar"
    path: "/bin/false"
  }
  tool_path {
    name: "compat-ld"
    path: "/bin/false"
  }
  tool_path {
    name: "cpp"
    path: "/bin/false"
  }
  tool_path {
    name: "dwp"
    path: "/bin/false"
  }
  tool_path {
    name: "gcc"
    path: "/bin/false"
  }
  tool_path {
    name: "gcov"
    path: "/bin/false"
  }
  tool_path {
    name: "ld"
    path: "/bin/false"
  }
  tool_path {
    name: "nm"
    path: "/bin/false"
  }
  tool_path {
    name: "objcopy"
    path: "/bin/false"
  }
  tool_path {
    name: "objdump"
    path: "/bin/false"
  }
  tool_path {
    name: "strip"
    path: "/bin/false"
  }
  builtin_sysroot: ""
  feature {
    name: "supports_dynamic_linker"
    enabled: true
  }
  feature {
    name: "supports_pic"
    enabled: true
  }
}
toolchain {
  toolchain_identifier: "%{toolchain_name}"
%{top_level_content}
  feature {
    name: "default_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "lto-backend"
      action: "clif-match"
      flag_group {
%{compile_content}
      }
    }
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "lto-backend"
      action: "clif-match"
      flag_group {
%{dbg_compile_content}
      }
      with_feature {
        feature: "dbg"
      }
    }
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "lto-backend"
      action: "clif-match"
      flag_group {
%{opt_compile_content}
      }
      with_feature {
        feature: "opt"
      }
    }
    flag_set {
      action: "linkstamp-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "lto-backend"
      action: "clif-match"
      flag_group {
%{cxx_content}
      }
    }
    enabled: true
  }
  feature {
    name: "default_link_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
%{link_content}
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
%{opt_link_content}
      }
      with_feature {
        feature: "opt"
      }
    }
    enabled: true
  }
  feature {
    name: "fdo_optimize"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fprofile-use=%{fdo_profile_path}"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "supports_dynamic_linker"
    enabled: true
  }
  feature {
    name: "dbg"
  }
  feature {
    name: "opt"
  }
  feature {
    name: "user_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "lto-backend"
      action: "clif-match"
      flag_group {
        flag: "%{user_compile_flags}"
        iterate_over: "user_compile_flags"
        expand_if_all_available: "user_compile_flags"
      }
    }
    enabled: true
  }
  feature {
    name: "sysroot"
    flag_set {
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "lto-backend"
      action: "clif-match"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "--sysroot=%{sysroot}"
        expand_if_all_available: "sysroot"
      }
    }
    enabled: true
  }
  feature {
    name: "unfiltered_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "lto-backend"
      action: "clif-match"
      flag_group {
%{unfiltered_content}
      }
    }
    enabled: true
  }
}
toolchain {
  toolchain_identifier: "msys_x64_mingw"
  host_system_name: "local"
  target_system_name: "local"
  target_cpu: "x64_windows"
  target_libc: "mingw"
  compiler: "mingw-gcc"
  abi_version: "local"
  abi_libc_version: "local"
  builtin_sysroot: ""
%{msys_x64_mingw_top_level_content}
  feature {
    name: "default_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "lto-backend"
      action: "clif-match"
      flag_group {
      }
    }
    flag_set {
      action: "linkstamp-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "lto-backend"
      action: "clif-match"
      flag_group {
%{msys_x64_mingw_cxx_content}
      }
    }
    enabled: true
  }
  feature {
    name: "default_link_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
%{msys_x64_mingw_link_content}
      }
    }
    enabled: true
  }
  feature {
    name: "supports_dynamic_linker"
    enabled: true
  }
  artifact_name_pattern {
    category_name: "executable"
    prefix: ""
    extension: ".exe"
  }
}
toolchain {
  toolchain_identifier: "msvc_x64"
  host_system_name: "local"
  target_system_name: "local"
  target_cpu: "x64_windows"
  target_libc: "msvcrt"
  compiler: "msvc-cl"
  abi_version: "local"
  abi_libc_version: "local"
%{msvc_x64_top_level_content}
  tool_path {
    name: "ar"
    path: "%{msvc_lib_path}"
  }
  tool_path {
    name: "ml"
    path: "%{msvc_ml_path}"
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
  feature {
    name: "no_legacy_features"
  }
  feature {
    name: "nologo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "/nologo"
      }
    }
  }
  feature {
    name: "has_configured_linker_path"
  }
  feature {
    name: "no_stripping"
  }
  feature {
    name: "targets_windows"
    implies: "copy_dynamic_libraries_to_binary"
    enabled: true
  }
  feature {
    name: "copy_dynamic_libraries_to_binary"
  }
  feature {
    name: "default_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "lto-backend"
      action: "clif-match"
      flag_group {
        flag: "/DCOMPILER_MSVC"
        flag: "/DNOMINMAX"
        flag: "/D_WIN32_WINNT=0x0601"
        flag: "/D_CRT_SECURE_NO_DEPRECATE"
        flag: "/D_CRT_SECURE_NO_WARNINGS"
        flag: "/bigobj"
        flag: "/Zm500"
        flag: "/EHsc"
        flag: "/wd4351"
        flag: "/wd4291"
        flag: "/wd4250"
        flag: "/wd4996"
      }
    }
    enabled: true
  }
  feature {
    name: "msvc_env"
    implies: "msvc_compile_env"
    implies: "msvc_link_env"
    env_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      env_entry {
        key: "PATH"
        value: "%{msvc_env_path}"
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
    name: "msvc_compile_env"
    env_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      env_entry {
        key: "INCLUDE"
        value: "%{msvc_env_include}"
      }
    }
  }
  feature {
    name: "msvc_link_env"
    env_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      env_entry {
        key: "LIB"
        value: "%{msvc_env_lib}"
      }
    }
  }
  feature {
    name: "include_paths"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      flag_group {
        flag: "/I%{quote_include_paths}"
        iterate_over: "quote_include_paths"
      }
      flag_group {
        flag: "/I%{include_paths}"
        iterate_over: "include_paths"
      }
      flag_group {
        flag: "/I%{system_include_paths}"
        iterate_over: "system_include_paths"
      }
    }
  }
  feature {
    name: "preprocessor_defines"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      flag_group {
        flag: "/D%{preprocessor_defines}"
        iterate_over: "preprocessor_defines"
      }
    }
  }
  feature {
    name: "parse_showincludes"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      flag_group {
        flag: "/showIncludes"
      }
    }
  }
  feature {
    name: "generate_pdb_file"
    requires {
      feature: "dbg"
    }
    requires {
      feature: "fastbuild"
    }
  }
  feature {
    name: "shared_flag"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "/DLL"
      }
    }
  }
  feature {
    name: "linkstamps"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "%{linkstamp_paths}"
        iterate_over: "linkstamp_paths"
        expand_if_all_available: "linkstamp_paths"
      }
    }
  }
  feature {
    name: "output_execpath_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "/OUT:%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "archiver_flags"
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "/OUT:%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "input_param_flags"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "/IMPLIB:%{interface_library_output_path}"
        expand_if_all_available: "interface_library_output_path"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "%{libopts}"
        iterate_over: "libopts"
        expand_if_all_available: "libopts"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.object_files}"
          }
          iterate_over: "libraries_to_link.object_files"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "interface_library"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "/WHOLEARCHIVE:%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "static_library"
          }
        }
        iterate_over: "libraries_to_link"
        expand_if_all_available: "libraries_to_link"
      }
    }
  }
  feature {
    name: "linker_subsystem_flag"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "/SUBSYSTEM:CONSOLE"
      }
    }
  }
  feature {
    name: "user_link_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "%{user_link_flags}"
        iterate_over: "user_link_flags"
        expand_if_all_available: "user_link_flags"
      }
    }
  }
  feature {
    name: "default_link_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "/MACHINE:X64"
      }
    }
    enabled: true
  }
  feature {
    name: "linker_param_file"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
      }
    }
  }
  feature {
    name: "static_link_msvcrt"
  }
  feature {
    name: "static_link_msvcrt_no_debug"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "/MT"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "/DEFAULTLIB:libcmt.lib"
      }
    }
    requires {
      feature: "fastbuild"
    }
    requires {
      feature: "opt"
    }
  }
  feature {
    name: "dynamic_link_msvcrt_no_debug"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "/MD"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "/DEFAULTLIB:msvcrt.lib"
      }
    }
    requires {
      feature: "fastbuild"
    }
    requires {
      feature: "opt"
    }
  }
  feature {
    name: "static_link_msvcrt_debug"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "/MTd"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "/DEFAULTLIB:libcmtd.lib"
      }
    }
    requires {
      feature: "dbg"
    }
  }
  feature {
    name: "dynamic_link_msvcrt_debug"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "/MDd"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "/DEFAULTLIB:msvcrtd.lib"
      }
    }
    requires {
      feature: "dbg"
    }
  }
  feature {
    name: "dbg"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "/Od"
        flag: "/Z7"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "%{dbg_mode_debug}"
        flag: "/INCREMENTAL:NO"
      }
    }
    implies: "generate_pdb_file"
  }
  feature {
    name: "fastbuild"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "/Od"
        flag: "/Z7"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "%{fastbuild_mode_debug}"
        flag: "/INCREMENTAL:NO"
      }
    }
    implies: "generate_pdb_file"
  }
  feature {
    name: "opt"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "/O2"
      }
    }
    implies: "frame_pointer"
  }
  feature {
    name: "frame_pointer"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "/Oy-"
      }
    }
  }
  feature {
    name: "disable_assertions"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "/DNDEBUG"
      }
      with_feature {
        feature: "opt"
      }
    }
    enabled: true
  }
  feature {
    name: "determinism"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "/wd4117"
        flag: "-D__DATE__=\"redacted\""
        flag: "-D__TIMESTAMP__=\"redacted\""
        flag: "-D__TIME__=\"redacted\""
      }
    }
    enabled: true
  }
  feature {
    name: "treat_warnings_as_errors"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "/WX"
      }
    }
  }
  feature {
    name: "smaller_binary"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "/Gy"
        flag: "/Gw"
      }
      with_feature {
        feature: "opt"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "/OPT:ICF"
        flag: "/OPT:REF"
      }
      with_feature {
        feature: "opt"
      }
    }
    enabled: true
  }
  feature {
    name: "ignore_noisy_warnings"
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "/ignore:4221"
      }
    }
    enabled: true
  }
  feature {
    name: "user_compile_flags"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      flag_group {
        flag: "%{user_compile_flags}"
        iterate_over: "user_compile_flags"
        expand_if_all_available: "user_compile_flags"
      }
    }
  }
  feature {
    name: "sysroot"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "--sysroot=%{sysroot}"
        iterate_over: "sysroot"
        expand_if_all_available: "sysroot"
      }
    }
  }
  feature {
    name: "unfiltered_compile_flags"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      flag_group {
        flag: "%{unfiltered_compile_flags}"
        iterate_over: "unfiltered_compile_flags"
        expand_if_all_available: "unfiltered_compile_flags"
      }
    }
  }
  feature {
    name: "compiler_output_flags"
    flag_set {
      action: "assemble"
      flag_group {
        flag: "/Fo%{output_file}"
        flag: "/Zi"
        expand_if_all_available: "output_file"
        expand_if_none_available: "output_assembly_file"
        expand_if_none_available: "output_preprocess_file"
      }
    }
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      flag_group {
        flag: "/Fo%{output_file}"
        expand_if_all_available: "output_file"
        expand_if_none_available: "output_assembly_file"
        expand_if_none_available: "output_preprocess_file"
      }
      flag_group {
        flag: "/Fa%{output_file}"
        expand_if_all_available: "output_file"
        expand_if_all_available: "output_assembly_file"
      }
      flag_group {
        flag: "/P"
        flag: "/Fi%{output_file}"
        expand_if_all_available: "output_file"
        expand_if_all_available: "output_preprocess_file"
      }
    }
  }
  feature {
    name: "compiler_input_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      flag_group {
        flag: "/c"
        flag: "%{source_file}"
        expand_if_all_available: "source_file"
      }
    }
  }
  feature {
    name: "def_file"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "/DEF:%{def_file_path}"
        flag: "/ignore:4070"
        expand_if_all_available: "def_file_path"
      }
    }
  }
  feature {
    name: "windows_export_all_symbols"
  }
  feature {
    name: "no_windows_export_all_symbols"
  }
  feature {
    name: "supports_dynamic_linker"
    enabled: true
  }
  feature {
    name: "supports_interface_shared_libraries"
    enabled: true
  }
  action_config {
    config_name: "assemble"
    action_name: "assemble"
    tool {
      tool_path: "%{msvc_ml_path}"
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "nologo"
    implies: "msvc_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "preprocess-assemble"
    action_name: "preprocess-assemble"
    tool {
      tool_path: "%{msvc_ml_path}"
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "nologo"
    implies: "msvc_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c-compile"
    action_name: "c-compile"
    tool {
      tool_path: "%{msvc_cl_path}"
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "default_compile_flags"
    implies: "nologo"
    implies: "msvc_env"
    implies: "parse_showincludes"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
  }
  action_config {
    config_name: "c++-compile"
    action_name: "c++-compile"
    tool {
      tool_path: "%{msvc_cl_path}"
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "default_compile_flags"
    implies: "nologo"
    implies: "msvc_env"
    implies: "parse_showincludes"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
  }
  action_config {
    config_name: "c++-link-executable"
    action_name: "c++-link-executable"
    tool {
      tool_path: "%{msvc_link_path}"
    }
    implies: "nologo"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "input_param_flags"
    implies: "user_link_flags"
    implies: "default_link_flags"
    implies: "linker_subsystem_flag"
    implies: "linker_param_file"
    implies: "msvc_env"
    implies: "no_stripping"
  }
  action_config {
    config_name: "c++-link-dynamic-library"
    action_name: "c++-link-dynamic-library"
    tool {
      tool_path: "%{msvc_link_path}"
    }
    implies: "nologo"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "input_param_flags"
    implies: "user_link_flags"
    implies: "default_link_flags"
    implies: "linker_subsystem_flag"
    implies: "linker_param_file"
    implies: "msvc_env"
    implies: "no_stripping"
    implies: "has_configured_linker_path"
    implies: "def_file"
  }
  action_config {
    config_name: "c++-link-nodeps-dynamic-library"
    action_name: "c++-link-nodeps-dynamic-library"
    tool {
      tool_path: "%{msvc_link_path}"
    }
    implies: "nologo"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "input_param_flags"
    implies: "user_link_flags"
    implies: "default_link_flags"
    implies: "linker_subsystem_flag"
    implies: "linker_param_file"
    implies: "msvc_env"
    implies: "no_stripping"
    implies: "has_configured_linker_path"
    implies: "def_file"
  }
  action_config {
    config_name: "c++-link-static-library"
    action_name: "c++-link-static-library"
    tool {
      tool_path: "%{msvc_lib_path}"
    }
    implies: "nologo"
    implies: "archiver_flags"
    implies: "input_param_flags"
    implies: "linker_param_file"
    implies: "msvc_env"
  }
  artifact_name_pattern {
    category_name: "object_file"
    prefix: ""
    extension: ".obj"
  }
  artifact_name_pattern {
    category_name: "static_library"
    prefix: ""
    extension: ".lib"
  }
  artifact_name_pattern {
    category_name: "alwayslink_static_library"
    prefix: ""
    extension: ".lo.lib"
  }
  artifact_name_pattern {
    category_name: "executable"
    prefix: ""
    extension: ".exe"
  }
  artifact_name_pattern {
    category_name: "dynamic_library"
    prefix: ""
    extension: ".dll"
  }
  artifact_name_pattern {
    category_name: "interface_library"
    prefix: ""
    extension: ".if.lib"
  }
}

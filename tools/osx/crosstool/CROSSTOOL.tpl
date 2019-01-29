major_version: "1"
minor_version: "0"
toolchain {
  toolchain_identifier: "darwin_x86_64"
  host_system_name: "x86_64-apple-macosx"
  target_system_name: "x86_64-apple-macosx"
  target_cpu: "darwin_x86_64"
  target_libc: "macosx"
  compiler: "compiler"
  abi_version: "darwin_x86_64"
  abi_libc_version: "darwin_x86_64"
  tool_path {
    name: "ar"
    path: "wrapped_ar"
  }
  tool_path {
    name: "compat-ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "cpp"
    path: "/usr/bin/cpp"
  }
  tool_path {
    name: "dwp"
    path: "/usr/bin/dwp"
  }
  tool_path {
    name: "gcc"
    path: "cc_wrapper.sh"
  }
  tool_path {
    name: "gcov"
    path: "/usr/bin/gcov"
  }
  tool_path {
    name: "ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "nm"
    path: "/usr/bin/nm"
  }
  tool_path {
    name: "objcopy"
    path: "/usr/bin/objcopy"
  }
  tool_path {
    name: "objdump"
    path: "/usr/bin/objdump"
  }
  tool_path {
    name: "strip"
    path: "/usr/bin/strip"
  }
  make_variable {
    name: "STACK_FRAME_UNLIMITED"
    value: "-Wframe-larger-than=100000000 -Wno-vla"
  }
  %{cxx_builtin_include_directory}
  builtin_sysroot: ""
  feature {
    name: "fastbuild"
  }
  feature {
    name: "no_legacy_features"
  }
  feature {
    name: "opt"
  }
  feature {
    name: "dbg"
  }
  feature {
    name: "link_libc++"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-lc++"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    enabled: true
  }
  feature {
    name: "compile_all_modules"
  }
  feature {
    name: "exclude_private_headers_in_module_maps"
  }
  feature {
    name: "has_configured_linker_path"
  }
  feature {
    name: "only_doth_headers_in_module_maps"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-fstack-protector"
        flag: "-fcolor-diagnostics"
        flag: "-Wall"
        flag: "-Wthread-safety"
        flag: "-Wself-assign"
        flag: "-fno-omit-frame-pointer"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-O0"
        flag: "-DDEBUG"
      }
      with_feature {
        feature: "fastbuild"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g0"
        flag: "-O2"
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-DNDEBUG"
        flag: "-ffunction-sections"
        flag: "-fdata-sections"
      }
      with_feature {
        feature: "opt"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g"
      }
      with_feature {
        feature: "dbg"
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
        flag: "-std=c++11"
      }
    }
    enabled: true
  }
  feature {
    name: "debug_prefix_map_pwd_is_dot"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "DEBUG_PREFIX_MAP_PWD=."
      }
    }
  }
  feature {
    name: "generate_dsym_file"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-g"
      }
    }
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "DSYM_HINT_LINKED_BINARY=%{linked_binary}"
        flag: "DSYM_HINT_DSYM_PATH=%{dsym_path}"
      }
    }
  }
  feature {
    name: "contains_objc_source"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fobjc-link-runtime"
      }
    }
  }
  feature {
    name: "objc_actions"
    implies: "objc-compile"
    implies: "objc++-compile"
    implies: "objc-fully-link"
    implies: "objc-archive"
    implies: "objc-executable"
    implies: "objc++-executable"
    implies: "assemble"
    implies: "preprocess-assemble"
    implies: "c-compile"
    implies: "c++-compile"
    implies: "c++-link-static-library"
    implies: "c++-link-dynamic-library"
    implies: "c++-link-nodeps-dynamic-library"
    implies: "c++-link-executable"
  }
  feature {
    name: "strip_debug_symbols"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-Wl,-S"
        expand_if_all_available: "strip_debug_symbols"
      }
    }
  }
  feature {
    name: "symbol_counts"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,--print-symbol-counts=%{symbol_counts_output}"
        expand_if_all_available: "symbol_counts_output"
      }
    }
  }
  feature {
    name: "shared_flag"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-shared"
      }
    }
  }
  feature {
    name: "kernel_extension"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-nostdlib"
        flag: "-lkmod"
        flag: "-lkmodc++"
        flag: "-lcc_kext"
        flag: "-Xlinker"
        flag: "-kext"
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
        flag: "-o"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "archiver_flags"
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "rcS"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "runtime_root_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-Wl,-rpath,@loader_path/%{runtime_library_search_directories}"
        iterate_over: "runtime_library_search_directories"
        expand_if_all_available: "runtime_library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_flags}"
        iterate_over: "runtime_root_flags"
        expand_if_all_available: "runtime_root_flags"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_entries}"
        iterate_over: "runtime_root_entries"
        expand_if_all_available: "runtime_root_entries"
      }
    }
  }
  feature {
    name: "input_param_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-L%{library_search_directories}"
        iterate_over: "library_search_directories"
        expand_if_all_available: "library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
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
        flag: "-Wl,-force_load,%{whole_archive_linker_params}"
        iterate_over: "whole_archive_linker_params"
        expand_if_all_available: "whole_archive_linker_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{linker_input_params}"
        iterate_over: "linker_input_params"
        expand_if_all_available: "linker_input_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag_group {
          flag: "-Wl,--start-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.object_files}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.object_files}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          iterate_over: "libraries_to_link.object_files"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag: "-Wl,--end-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
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
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "static_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "dynamic_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l:%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l:%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "versioned_dynamic_library"
          }
        }
        iterate_over: "libraries_to_link"
        expand_if_all_available: "libraries_to_link"
      }
    }
  }
  feature {
    name: "force_pic_flags"
    flag_set {
      action: "c++-link-executable"
      flag_group {
        flag: "-Wl,-pie"
        expand_if_all_available: "force_pic"
      }
    }
  }
  feature {
    name: "pch"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-include"
        flag: "%{pch_file}"
      }
    }
  }
  feature {
    name: "module_maps"
  }
  feature {
    name: "use_objc_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-name=%{module_name}"
        flag: "-iquote"
        flag: "%{module_maps_dir}"
        flag: "-fmodules-cache-path=%{modules_cache_path}"
      }
    }
  }
  feature {
    name: "no_enable_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-maps"
      }
    }
    requires {
      feature: "use_objc_modules"
    }
  }
  feature {
    name: "apply_default_warnings"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-Wshorten-64-to-32"
        flag: "-Wbool-conversion"
        flag: "-Wconstant-conversion"
        flag: "-Wduplicate-method-match"
        flag: "-Wempty-body"
        flag: "-Wenum-conversion"
        flag: "-Wint-conversion"
        flag: "-Wunreachable-code"
        flag: "-Wmismatched-return-types"
        flag: "-Wundeclared-selector"
        flag: "-Wuninitialized"
        flag: "-Wunused-function"
        flag: "-Wunused-variable"
      }
    }
  }
  feature {
    name: "includes"
    flag_set {
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "clif-match"
      flag_group {
        flag: "-include"
        flag: "%{includes}"
        iterate_over: "includes"
        expand_if_all_available: "includes"
      }
    }
    enabled: true
  }
  feature {
    name: "include_paths"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "clif-match"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-iquote"
        flag: "%{quote_include_paths}"
        iterate_over: "quote_include_paths"
      }
      flag_group {
        flag: "-I%{include_paths}"
        iterate_over: "include_paths"
      }
      flag_group {
        flag: "-isystem"
        flag: "%{system_include_paths}"
        iterate_over: "system_include_paths"
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
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "linkstamp-compile"
      action: "clif-match"
      flag_group {
        flag: "--sysroot=%{sysroot}"
        expand_if_all_available: "sysroot"
      }
    }
  }
  feature {
    name: "dependency_file"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      flag_group {
        flag: "-MD"
        flag: "-MF"
        flag: "%{dependency_file}"
        expand_if_all_available: "dependency_file"
      }
    }
  }
  feature {
    name: "pic"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "preprocess-assemble"
      flag_group {
        flag: "-fPIC"
        expand_if_all_available: "pic"
      }
    }
  }
  feature {
    name: "per_object_debug_info"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-gsplit-dwarf"
        expand_if_all_available: "per_object_debug_info_file"
      }
    }
  }
  feature {
    name: "preprocessor_defines"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D%{preprocessor_defines}"
        iterate_over: "preprocessor_defines"
      }
    }
  }
  feature {
    name: "framework_paths"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-F%{framework_paths}"
        iterate_over: "framework_paths"
      }
    }
  }
  feature {
    name: "random_seed"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      flag_group {
        flag: "-frandom-seed=%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "fdo_instrument"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fprofile-generate=%{fdo_instrument_path}"
        flag: "-fno-data-sections"
        expand_if_all_available: "fdo_instrument_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "fdo_optimize"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fprofile-use=%{fdo_profile_path}"
        flag: "-Xclang-only=-Wno-profile-instr-unprofiled"
        flag: "-Xclang-only=-Wno-profile-instr-out-of-date"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "autofdo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fauto-profile=%{fdo_profile_path}"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "lipo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fripa"
      }
    }
    requires {
      feature: "autofdo"
    }
    requires {
      feature: "fdo_optimize"
    }
    requires {
      feature: "fdo_instrument"
    }
  }
  feature {
    name: "coverage"
  }
  feature {
    name: "llvm_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-instr-generate"
        flag: "-fcoverage-mapping"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-fprofile-instr-generate"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "gcc_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-arcs"
        flag: "-ftest-coverage"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "--coverage"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "apply_default_compiler_flags"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-DOS_MACOSX"
        flag: "-fno-autolink"
      }
    }
  }
  feature {
    name: "include_system_dirs"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-isysroot"
        flag: "%{sdk_dir}"
      }
    }
  }
  feature {
    name: "bitcode_embedded"
  }
  feature {
    name: "bitcode_embedded_markers"
  }
  feature {
    name: "objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fobjc-arc"
        expand_if_all_available: "objc_arc"
      }
    }
  }
  feature {
    name: "no_objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fno-objc-arc"
        expand_if_all_available: "no_objc_arc"
      }
    }
  }
  feature {
    name: "apple_env"
    env_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-archive"
      action: "objc-fully-link"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      action: "objc-executable"
      action: "objc++-executable"
      action: "linkstamp-compile"
      env_entry {
        key: "XCODE_VERSION_OVERRIDE"
        value: "%{xcode_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_VERSION_OVERRIDE"
        value: "%{apple_sdk_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_PLATFORM"
        value: "%{apple_sdk_platform_value}"
      }
    }
  }
  feature {
    name: "user_link_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "%{user_link_flags}"
        iterate_over: "user_link_flags"
        expand_if_all_available: "user_link_flags"
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
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-headerpad_max_install_names"
        flag: "-no-canonical-prefixes"
      }
    }
    flag_set {
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-undefined"
        flag: "dynamic_lookup"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-undefined"
        flag: "dynamic_lookup"
      }
      with_feature {
        feature: "dynamic_linking_mode"
      }
    }
    enabled: true
  }
  feature {
    name: "version_min"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-mmacosx-version-min=%{version_min}"
      }
    }
  }
  feature {
    name: "dead_strip"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-dead_strip"
        flag: "-no_dead_strip_inits_and_terms"
      }
    }
    requires {
      feature: "opt"
    }
  }
  feature {
    name: "cpp_linker_flags"
  }
  feature {
    name: "apply_implicit_frameworks"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-framework"
        flag: "Foundation"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
  }
  feature {
    name: "link_cocoa"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-framework"
        flag: "Cocoa"
      }
    }
  }
  feature {
    name: "apply_simulator_compiler_flags"
  }
  feature {
    name: "unfiltered_cxx_flags"
  }
  feature {
    name: "user_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "%{user_compile_flags}"
        iterate_over: "user_compile_flags"
        expand_if_all_available: "user_compile_flags"
      }
    }
  }
  feature {
    name: "unfiltered_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      flag_group {
        flag: "-no-canonical-prefixes"
        flag: "-Wno-builtin-macro-redefined"
        flag: "-D__DATE__=\"redacted\""
        flag: "-D__TIMESTAMP__=\"redacted\""
        flag: "-D__TIME__=\"redacted\""
      }
    }
  }
  feature {
    name: "linker_param_file"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
      }
    }
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
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
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-c"
        flag: "%{source_file}"
        expand_if_all_available: "source_file"
      }
    }
  }
  feature {
    name: "compiler_output_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-S"
        expand_if_all_available: "output_assembly_file"
      }
      flag_group {
        flag: "-E"
        expand_if_all_available: "output_preprocess_file"
      }
      flag_group {
        flag: "-o"
        flag: "%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "supports_dynamic_linker"
    enabled: true
  }
  feature {
    name: "objcopy_embed_flags"
    flag_set {
      action: "objcopy_embed_data"
      flag_group {
        flag: "-I"
        flag: "binary"
      }
    }
    enabled: true
  }
  feature {
    name: "dynamic_linking_mode"
  }
  action_config {
    config_name: "strip"
    action_name: "strip"
    tool {
      tool_path: "/usr/bin/strip"
    }
    flag_set {
      flag_group {
        flag: "-S"
        flag: "-o"
        flag: "%{output_file}"
      }
      flag_group {
        flag: "%{stripopts}"
        iterate_over: "stripopts"
      }
      flag_group {
        flag: "%{input_file}"
      }
    }
  }
  action_config {
    config_name: "c-compile"
    action_name: "c-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-compile"
    action_name: "c++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "linkstamp-compile"
    action_name: "linkstamp-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-module-compile"
    action_name: "c++-module-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-header-parsing"
    action_name: "c++-header-parsing"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-compile"
    action_name: "objc-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "x86_64"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "objc_actions"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
  }
  action_config {
    config_name: "objc++-compile"
    action_name: "objc++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "x86_64"
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
  }
  action_config {
    config_name: "assemble"
    action_name: "assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "include_system_dirs"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "preprocess-assemble"
    action_name: "preprocess-assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-archive"
    action_name: "objc-archive"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-filelist"
        flag: "%{obj_list_path}"
        flag: "-arch_only"
        flag: "x86_64"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{archive_path}"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-executable"
    action_name: "objc-executable"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "x86_64"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "objc++-executable"
    action_name: "objc++-executable"
    tool {
      tool_path: "wrapped_clang_pp"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
      flag_group {
        flag: "-arch"
        flag: "x86_64"
      }
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "c++-link-executable"
    action_name: "c++-link-executable"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "symbol_counts"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "force_pic_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-dynamic-library"
    action_name: "c++-link-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-nodeps-dynamic-library"
    action_name: "c++-link-nodeps-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-static-library"
    action_name: "c++-link-static-library"
    tool {
      tool_path: "wrapped_ar"
      execution_requirement: "requires-darwin"
    }
    implies: "runtime_root_flags"
    implies: "archiver_flags"
    implies: "input_param_flags"
    implies: "linker_param_file"
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-fully-link"
    action_name: "objc-fully-link"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-arch_only"
        flag: "x86_64"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{fully_linked_archive_path}"
      }
      flag_group {
        flag: "%{objc_library_exec_paths}"
        iterate_over: "objc_library_exec_paths"
      }
      flag_group {
        flag: "%{cc_library_exec_paths}"
        iterate_over: "cc_library_exec_paths"
      }
      flag_group {
        flag: "%{imported_library_exec_paths}"
        iterate_over: "imported_library_exec_paths"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objcopy_embed_data"
    action_name: "objcopy_embed_data"
    tool {
      tool_path: "/usr/bin/objcopy"
    }
    enabled: true
  }
  cc_target_os: "apple"
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
  tool_path {
    name: "ar"
    path: "wrapped_ar"
  }
  tool_path {
    name: "compat-ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "cpp"
    path: "/usr/bin/cpp"
  }
  tool_path {
    name: "dwp"
    path: "/usr/bin/dwp"
  }
  tool_path {
    name: "gcc"
    path: "cc_wrapper.sh"
  }
  tool_path {
    name: "gcov"
    path: "/usr/bin/gcov"
  }
  tool_path {
    name: "ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "nm"
    path: "/usr/bin/nm"
  }
  tool_path {
    name: "objcopy"
    path: "/usr/bin/objcopy"
  }
  tool_path {
    name: "objdump"
    path: "/usr/bin/objdump"
  }
  tool_path {
    name: "strip"
    path: "/usr/bin/strip"
  }
  make_variable {
    name: "STACK_FRAME_UNLIMITED"
    value: "-Wframe-larger-than=100000000 -Wno-vla"
  }
  %{cxx_builtin_include_directory}
  builtin_sysroot: ""
  feature {
    name: "fastbuild"
  }
  feature {
    name: "no_legacy_features"
  }
  feature {
    name: "opt"
  }
  feature {
    name: "dbg"
  }
  feature {
    name: "link_libc++"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-lc++"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    enabled: true
  }
  feature {
    name: "compile_all_modules"
  }
  feature {
    name: "exclude_private_headers_in_module_maps"
  }
  feature {
    name: "has_configured_linker_path"
  }
  feature {
    name: "only_doth_headers_in_module_maps"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-fstack-protector"
        flag: "-fcolor-diagnostics"
        flag: "-Wall"
        flag: "-Wthread-safety"
        flag: "-Wself-assign"
        flag: "-fno-omit-frame-pointer"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-O0"
        flag: "-DDEBUG"
      }
      with_feature {
        feature: "fastbuild"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g0"
        flag: "-O2"
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-DNDEBUG"
        flag: "-ffunction-sections"
        flag: "-fdata-sections"
      }
      with_feature {
        feature: "opt"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g"
      }
      with_feature {
        feature: "dbg"
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
        flag: "-std=c++11"
      }
    }
    enabled: true
  }
  feature {
    name: "debug_prefix_map_pwd_is_dot"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "DEBUG_PREFIX_MAP_PWD=."
      }
    }
  }
  feature {
    name: "generate_dsym_file"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-g"
      }
    }
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "DSYM_HINT_LINKED_BINARY=%{linked_binary}"
        flag: "DSYM_HINT_DSYM_PATH=%{dsym_path}"
      }
    }
  }
  feature {
    name: "contains_objc_source"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fobjc-link-runtime"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "objc_actions"
    implies: "objc-compile"
    implies: "objc++-compile"
    implies: "objc-fully-link"
    implies: "objc-archive"
    implies: "objc-executable"
    implies: "objc++-executable"
    implies: "assemble"
    implies: "preprocess-assemble"
    implies: "c-compile"
    implies: "c++-compile"
    implies: "c++-link-static-library"
    implies: "c++-link-dynamic-library"
    implies: "c++-link-nodeps-dynamic-library"
    implies: "c++-link-executable"
  }
  feature {
    name: "strip_debug_symbols"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-Wl,-S"
        expand_if_all_available: "strip_debug_symbols"
      }
    }
  }
  feature {
    name: "symbol_counts"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,--print-symbol-counts=%{symbol_counts_output}"
        expand_if_all_available: "symbol_counts_output"
      }
    }
  }
  feature {
    name: "shared_flag"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-shared"
      }
    }
  }
  feature {
    name: "kernel_extension"
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
        flag: "-o"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "archiver_flags"
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "rcS"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "runtime_root_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-Wl,-rpath,@loader_path/%{runtime_library_search_directories}"
        iterate_over: "runtime_library_search_directories"
        expand_if_all_available: "runtime_library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_flags}"
        iterate_over: "runtime_root_flags"
        expand_if_all_available: "runtime_root_flags"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_entries}"
        iterate_over: "runtime_root_entries"
        expand_if_all_available: "runtime_root_entries"
      }
    }
  }
  feature {
    name: "input_param_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-L%{library_search_directories}"
        iterate_over: "library_search_directories"
        expand_if_all_available: "library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
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
        flag: "-Wl,-force_load,%{whole_archive_linker_params}"
        iterate_over: "whole_archive_linker_params"
        expand_if_all_available: "whole_archive_linker_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{linker_input_params}"
        iterate_over: "linker_input_params"
        expand_if_all_available: "linker_input_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag_group {
          flag: "-Wl,--start-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.object_files}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.object_files}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          iterate_over: "libraries_to_link.object_files"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag: "-Wl,--end-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
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
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "static_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "dynamic_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l:%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l:%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "versioned_dynamic_library"
          }
        }
        iterate_over: "libraries_to_link"
        expand_if_all_available: "libraries_to_link"
      }
    }
  }
  feature {
    name: "force_pic_flags"
    flag_set {
      action: "c++-link-executable"
      flag_group {
        flag: "-Wl,-pie"
        expand_if_all_available: "force_pic"
      }
    }
  }
  feature {
    name: "pch"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-include"
        flag: "%{pch_file}"
      }
    }
  }
  feature {
    name: "module_maps"
  }
  feature {
    name: "use_objc_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-name=%{module_name}"
        flag: "-iquote"
        flag: "%{module_maps_dir}"
        flag: "-fmodules-cache-path=%{modules_cache_path}"
      }
    }
  }
  feature {
    name: "no_enable_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-maps"
      }
    }
    requires {
      feature: "use_objc_modules"
    }
  }
  feature {
    name: "apply_default_warnings"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-Wshorten-64-to-32"
        flag: "-Wbool-conversion"
        flag: "-Wconstant-conversion"
        flag: "-Wduplicate-method-match"
        flag: "-Wempty-body"
        flag: "-Wenum-conversion"
        flag: "-Wint-conversion"
        flag: "-Wunreachable-code"
        flag: "-Wmismatched-return-types"
        flag: "-Wundeclared-selector"
        flag: "-Wuninitialized"
        flag: "-Wunused-function"
        flag: "-Wunused-variable"
      }
    }
  }
  feature {
    name: "includes"
    flag_set {
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "clif-match"
      flag_group {
        flag: "-include"
        flag: "%{includes}"
        iterate_over: "includes"
        expand_if_all_available: "includes"
      }
    }
    enabled: true
  }
  feature {
    name: "include_paths"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "clif-match"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-iquote"
        flag: "%{quote_include_paths}"
        iterate_over: "quote_include_paths"
      }
      flag_group {
        flag: "-I%{include_paths}"
        iterate_over: "include_paths"
      }
      flag_group {
        flag: "-isystem"
        flag: "%{system_include_paths}"
        iterate_over: "system_include_paths"
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
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "linkstamp-compile"
      action: "clif-match"
      flag_group {
        flag: "--sysroot=%{sysroot}"
        expand_if_all_available: "sysroot"
      }
    }
  }
  feature {
    name: "dependency_file"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      flag_group {
        flag: "-MD"
        flag: "-MF"
        flag: "%{dependency_file}"
        expand_if_all_available: "dependency_file"
      }
    }
  }
  feature {
    name: "pic"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "preprocess-assemble"
      flag_group {
        flag: "-fPIC"
        expand_if_all_available: "pic"
      }
    }
  }
  feature {
    name: "per_object_debug_info"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-gsplit-dwarf"
        expand_if_all_available: "per_object_debug_info_file"
      }
    }
  }
  feature {
    name: "preprocessor_defines"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D%{preprocessor_defines}"
        iterate_over: "preprocessor_defines"
      }
    }
  }
  feature {
    name: "framework_paths"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-F%{framework_paths}"
        iterate_over: "framework_paths"
      }
    }
  }
  feature {
    name: "random_seed"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      flag_group {
        flag: "-frandom-seed=%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "fdo_instrument"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fprofile-generate=%{fdo_instrument_path}"
        flag: "-fno-data-sections"
        expand_if_all_available: "fdo_instrument_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "fdo_optimize"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fprofile-use=%{fdo_profile_path}"
        flag: "-Xclang-only=-Wno-profile-instr-unprofiled"
        flag: "-Xclang-only=-Wno-profile-instr-out-of-date"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "autofdo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fauto-profile=%{fdo_profile_path}"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "lipo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fripa"
      }
    }
    requires {
      feature: "autofdo"
    }
    requires {
      feature: "fdo_optimize"
    }
    requires {
      feature: "fdo_instrument"
    }
  }
  feature {
    name: "coverage"
  }
  feature {
    name: "llvm_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-instr-generate"
        flag: "-fcoverage-mapping"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-fprofile-instr-generate"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "gcc_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-arcs"
        flag: "-ftest-coverage"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "--coverage"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "apply_default_compiler_flags"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-DOS_IOS"
        flag: "-fno-autolink"
      }
    }
  }
  feature {
    name: "include_system_dirs"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-isysroot"
        flag: "%{sdk_dir}"
      }
    }
  }
  feature {
    name: "bitcode_embedded"
  }
  feature {
    name: "bitcode_embedded_markers"
  }
  feature {
    name: "objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fobjc-arc"
        expand_if_all_available: "objc_arc"
      }
    }
  }
  feature {
    name: "no_objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fno-objc-arc"
        expand_if_all_available: "no_objc_arc"
      }
    }
  }
  feature {
    name: "apple_env"
    env_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-archive"
      action: "objc-fully-link"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      action: "objc-executable"
      action: "objc++-executable"
      action: "linkstamp-compile"
      env_entry {
        key: "XCODE_VERSION_OVERRIDE"
        value: "%{xcode_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_VERSION_OVERRIDE"
        value: "%{apple_sdk_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_PLATFORM"
        value: "%{apple_sdk_platform_value}"
      }
    }
  }
  feature {
    name: "user_link_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "%{user_link_flags}"
        iterate_over: "user_link_flags"
        expand_if_all_available: "user_link_flags"
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
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-headerpad_max_install_names"
        flag: "-no-canonical-prefixes"
        flag: "-target"
        flag: "x86_64-apple-ios"
      }
    }
    enabled: true
  }
  feature {
    name: "version_min"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-mios-simulator-version-min=%{version_min}"
      }
    }
  }
  feature {
    name: "dead_strip"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-dead_strip"
        flag: "-no_dead_strip_inits_and_terms"
      }
    }
    requires {
      feature: "opt"
    }
  }
  feature {
    name: "cpp_linker_flags"
  }
  feature {
    name: "apply_implicit_frameworks"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-framework"
        flag: "Foundation"
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "link_cocoa"
  }
  feature {
    name: "apply_simulator_compiler_flags"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fexceptions"
        flag: "-fasm-blocks"
        flag: "-fobjc-abi-version=2"
        flag: "-fobjc-legacy-dispatch"
      }
    }
  }
  feature {
    name: "unfiltered_cxx_flags"
  }
  feature {
    name: "user_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "%{user_compile_flags}"
        iterate_over: "user_compile_flags"
        expand_if_all_available: "user_compile_flags"
      }
    }
  }
  feature {
    name: "unfiltered_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      flag_group {
        flag: "-no-canonical-prefixes"
        flag: "-Wno-builtin-macro-redefined"
        flag: "-D__DATE__=\"redacted\""
        flag: "-D__TIMESTAMP__=\"redacted\""
        flag: "-D__TIME__=\"redacted\""
        flag: "-target"
        flag: "x86_64-apple-ios"
      }
    }
  }
  feature {
    name: "linker_param_file"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
      }
    }
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
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
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-c"
        flag: "%{source_file}"
        expand_if_all_available: "source_file"
      }
    }
  }
  feature {
    name: "compiler_output_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-S"
        expand_if_all_available: "output_assembly_file"
      }
      flag_group {
        flag: "-E"
        expand_if_all_available: "output_preprocess_file"
      }
      flag_group {
        flag: "-o"
        flag: "%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "objcopy_embed_flags"
    flag_set {
      action: "objcopy_embed_data"
      flag_group {
        flag: "-I"
        flag: "binary"
      }
    }
    enabled: true
  }
  action_config {
    config_name: "strip"
    action_name: "strip"
    tool {
      tool_path: "/usr/bin/strip"
    }
    flag_set {
      flag_group {
        flag: "-S"
        flag: "-o"
        flag: "%{output_file}"
      }
      flag_group {
        flag: "%{stripopts}"
        iterate_over: "stripopts"
      }
      flag_group {
        flag: "%{input_file}"
      }
    }
  }
  action_config {
    config_name: "c-compile"
    action_name: "c-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-compile"
    action_name: "c++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "linkstamp-compile"
    action_name: "linkstamp-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-module-compile"
    action_name: "c++-module-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-header-parsing"
    action_name: "c++-header-parsing"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-compile"
    action_name: "objc-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "x86_64"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "objc_actions"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "apply_simulator_compiler_flags"
  }
  action_config {
    config_name: "objc++-compile"
    action_name: "objc++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "x86_64"
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "apply_simulator_compiler_flags"
  }
  action_config {
    config_name: "assemble"
    action_name: "assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "include_system_dirs"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "preprocess-assemble"
    action_name: "preprocess-assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-archive"
    action_name: "objc-archive"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-filelist"
        flag: "%{obj_list_path}"
        flag: "-arch_only"
        flag: "x86_64"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{archive_path}"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-executable"
    action_name: "objc-executable"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "x86_64"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "objc++-executable"
    action_name: "objc++-executable"
    tool {
      tool_path: "wrapped_clang_pp"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
      flag_group {
        flag: "-arch"
        flag: "x86_64"
      }
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "c++-link-executable"
    action_name: "c++-link-executable"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "symbol_counts"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "force_pic_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-dynamic-library"
    action_name: "c++-link-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-nodeps-dynamic-library"
    action_name: "c++-link-nodeps-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-static-library"
    action_name: "c++-link-static-library"
    tool {
      tool_path: "wrapped_ar"
      execution_requirement: "requires-darwin"
    }
    implies: "runtime_root_flags"
    implies: "archiver_flags"
    implies: "input_param_flags"
    implies: "linker_param_file"
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-fully-link"
    action_name: "objc-fully-link"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-arch_only"
        flag: "x86_64"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{fully_linked_archive_path}"
      }
      flag_group {
        flag: "%{objc_library_exec_paths}"
        iterate_over: "objc_library_exec_paths"
      }
      flag_group {
        flag: "%{cc_library_exec_paths}"
        iterate_over: "cc_library_exec_paths"
      }
      flag_group {
        flag: "%{imported_library_exec_paths}"
        iterate_over: "imported_library_exec_paths"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objcopy_embed_data"
    action_name: "objcopy_embed_data"
    tool {
      tool_path: "/usr/bin/objcopy"
    }
    enabled: true
  }
  cc_target_os: "apple"
}
toolchain {
  toolchain_identifier: "watchos_i386"
  host_system_name: "x86_64-apple-macosx"
  target_system_name: "i386-apple-watchos"
  target_cpu: "watchos_i386"
  target_libc: "watchos"
  compiler: "compiler"
  abi_version: "local"
  abi_libc_version: "local"
  tool_path {
    name: "ar"
    path: "wrapped_ar"
  }
  tool_path {
    name: "compat-ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "cpp"
    path: "/usr/bin/cpp"
  }
  tool_path {
    name: "dwp"
    path: "/usr/bin/dwp"
  }
  tool_path {
    name: "gcc"
    path: "cc_wrapper.sh"
  }
  tool_path {
    name: "gcov"
    path: "/usr/bin/gcov"
  }
  tool_path {
    name: "ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "nm"
    path: "/usr/bin/nm"
  }
  tool_path {
    name: "objcopy"
    path: "/usr/bin/objcopy"
  }
  tool_path {
    name: "objdump"
    path: "/usr/bin/objdump"
  }
  tool_path {
    name: "strip"
    path: "/usr/bin/strip"
  }
  make_variable {
    name: "STACK_FRAME_UNLIMITED"
    value: "-Wframe-larger-than=100000000 -Wno-vla"
  }
  %{cxx_builtin_include_directory}
  builtin_sysroot: ""
  feature {
    name: "fastbuild"
  }
  feature {
    name: "no_legacy_features"
  }
  feature {
    name: "opt"
  }
  feature {
    name: "dbg"
  }
  feature {
    name: "link_libc++"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-lc++"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    enabled: true
  }
  feature {
    name: "compile_all_modules"
  }
  feature {
    name: "exclude_private_headers_in_module_maps"
  }
  feature {
    name: "has_configured_linker_path"
  }
  feature {
    name: "only_doth_headers_in_module_maps"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-fstack-protector"
        flag: "-fcolor-diagnostics"
        flag: "-Wall"
        flag: "-Wthread-safety"
        flag: "-Wself-assign"
        flag: "-fno-omit-frame-pointer"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-O0"
        flag: "-DDEBUG"
      }
      with_feature {
        feature: "fastbuild"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g0"
        flag: "-O2"
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-DNDEBUG"
        flag: "-ffunction-sections"
        flag: "-fdata-sections"
      }
      with_feature {
        feature: "opt"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g"
      }
      with_feature {
        feature: "dbg"
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
        flag: "-std=c++11"
      }
    }
    enabled: true
  }
  feature {
    name: "debug_prefix_map_pwd_is_dot"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "DEBUG_PREFIX_MAP_PWD=."
      }
    }
  }
  feature {
    name: "generate_dsym_file"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-g"
      }
    }
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "DSYM_HINT_LINKED_BINARY=%{linked_binary}"
        flag: "DSYM_HINT_DSYM_PATH=%{dsym_path}"
      }
    }
  }
  feature {
    name: "contains_objc_source"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fobjc-link-runtime"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "objc_actions"
    implies: "objc-compile"
    implies: "objc++-compile"
    implies: "objc-fully-link"
    implies: "objc-archive"
    implies: "objc-executable"
    implies: "objc++-executable"
    implies: "assemble"
    implies: "preprocess-assemble"
    implies: "c-compile"
    implies: "c++-compile"
    implies: "c++-link-static-library"
    implies: "c++-link-dynamic-library"
    implies: "c++-link-nodeps-dynamic-library"
    implies: "c++-link-executable"
  }
  feature {
    name: "strip_debug_symbols"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-Wl,-S"
        expand_if_all_available: "strip_debug_symbols"
      }
    }
  }
  feature {
    name: "symbol_counts"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,--print-symbol-counts=%{symbol_counts_output}"
        expand_if_all_available: "symbol_counts_output"
      }
    }
  }
  feature {
    name: "shared_flag"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-shared"
      }
    }
  }
  feature {
    name: "kernel_extension"
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
        flag: "-o"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "archiver_flags"
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "rcS"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "runtime_root_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-Wl,-rpath,@loader_path/%{runtime_library_search_directories}"
        iterate_over: "runtime_library_search_directories"
        expand_if_all_available: "runtime_library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_flags}"
        iterate_over: "runtime_root_flags"
        expand_if_all_available: "runtime_root_flags"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_entries}"
        iterate_over: "runtime_root_entries"
        expand_if_all_available: "runtime_root_entries"
      }
    }
  }
  feature {
    name: "input_param_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-L%{library_search_directories}"
        iterate_over: "library_search_directories"
        expand_if_all_available: "library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
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
        flag: "-Wl,-force_load,%{whole_archive_linker_params}"
        iterate_over: "whole_archive_linker_params"
        expand_if_all_available: "whole_archive_linker_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{linker_input_params}"
        iterate_over: "linker_input_params"
        expand_if_all_available: "linker_input_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag_group {
          flag: "-Wl,--start-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.object_files}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.object_files}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          iterate_over: "libraries_to_link.object_files"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag: "-Wl,--end-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
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
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "static_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "dynamic_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l:%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l:%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "versioned_dynamic_library"
          }
        }
        iterate_over: "libraries_to_link"
        expand_if_all_available: "libraries_to_link"
      }
    }
  }
  feature {
    name: "force_pic_flags"
    flag_set {
      action: "c++-link-executable"
      flag_group {
        flag: "-Wl,-pie"
        expand_if_all_available: "force_pic"
      }
    }
  }
  feature {
    name: "pch"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-include"
        flag: "%{pch_file}"
      }
    }
  }
  feature {
    name: "module_maps"
  }
  feature {
    name: "use_objc_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-name=%{module_name}"
        flag: "-iquote"
        flag: "%{module_maps_dir}"
        flag: "-fmodules-cache-path=%{modules_cache_path}"
      }
    }
  }
  feature {
    name: "no_enable_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-maps"
      }
    }
    requires {
      feature: "use_objc_modules"
    }
  }
  feature {
    name: "apply_default_warnings"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-Wshorten-64-to-32"
        flag: "-Wbool-conversion"
        flag: "-Wconstant-conversion"
        flag: "-Wduplicate-method-match"
        flag: "-Wempty-body"
        flag: "-Wenum-conversion"
        flag: "-Wint-conversion"
        flag: "-Wunreachable-code"
        flag: "-Wmismatched-return-types"
        flag: "-Wundeclared-selector"
        flag: "-Wuninitialized"
        flag: "-Wunused-function"
        flag: "-Wunused-variable"
      }
    }
  }
  feature {
    name: "includes"
    flag_set {
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "clif-match"
      flag_group {
        flag: "-include"
        flag: "%{includes}"
        iterate_over: "includes"
        expand_if_all_available: "includes"
      }
    }
    enabled: true
  }
  feature {
    name: "include_paths"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "clif-match"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-iquote"
        flag: "%{quote_include_paths}"
        iterate_over: "quote_include_paths"
      }
      flag_group {
        flag: "-I%{include_paths}"
        iterate_over: "include_paths"
      }
      flag_group {
        flag: "-isystem"
        flag: "%{system_include_paths}"
        iterate_over: "system_include_paths"
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
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "linkstamp-compile"
      action: "clif-match"
      flag_group {
        flag: "--sysroot=%{sysroot}"
        expand_if_all_available: "sysroot"
      }
    }
  }
  feature {
    name: "dependency_file"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      flag_group {
        flag: "-MD"
        flag: "-MF"
        flag: "%{dependency_file}"
        expand_if_all_available: "dependency_file"
      }
    }
  }
  feature {
    name: "pic"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "preprocess-assemble"
      flag_group {
        flag: "-fPIC"
        expand_if_all_available: "pic"
      }
    }
  }
  feature {
    name: "per_object_debug_info"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-gsplit-dwarf"
        expand_if_all_available: "per_object_debug_info_file"
      }
    }
  }
  feature {
    name: "preprocessor_defines"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D%{preprocessor_defines}"
        iterate_over: "preprocessor_defines"
      }
    }
  }
  feature {
    name: "framework_paths"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-F%{framework_paths}"
        iterate_over: "framework_paths"
      }
    }
  }
  feature {
    name: "random_seed"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      flag_group {
        flag: "-frandom-seed=%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "fdo_instrument"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fprofile-generate=%{fdo_instrument_path}"
        flag: "-fno-data-sections"
        expand_if_all_available: "fdo_instrument_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "fdo_optimize"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fprofile-use=%{fdo_profile_path}"
        flag: "-Xclang-only=-Wno-profile-instr-unprofiled"
        flag: "-Xclang-only=-Wno-profile-instr-out-of-date"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "autofdo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fauto-profile=%{fdo_profile_path}"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "lipo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fripa"
      }
    }
    requires {
      feature: "autofdo"
    }
    requires {
      feature: "fdo_optimize"
    }
    requires {
      feature: "fdo_instrument"
    }
  }
  feature {
    name: "coverage"
  }
  feature {
    name: "llvm_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-instr-generate"
        flag: "-fcoverage-mapping"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-fprofile-instr-generate"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "gcc_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-arcs"
        flag: "-ftest-coverage"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "--coverage"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "apply_default_compiler_flags"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-DOS_IOS"
        flag: "-fno-autolink"
      }
    }
  }
  feature {
    name: "include_system_dirs"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-isysroot"
        flag: "%{sdk_dir}"
        flag: "-F%{sdk_framework_dir}"
        flag: "-F%{platform_developer_framework_dir}"
      }
    }
  }
  feature {
    name: "bitcode_embedded"
  }
  feature {
    name: "bitcode_embedded_markers"
  }
  feature {
    name: "objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fobjc-arc"
        expand_if_all_available: "objc_arc"
      }
    }
  }
  feature {
    name: "no_objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fno-objc-arc"
        expand_if_all_available: "no_objc_arc"
      }
    }
  }
  feature {
    name: "apple_env"
    env_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-archive"
      action: "objc-fully-link"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      action: "objc-executable"
      action: "objc++-executable"
      action: "linkstamp-compile"
      env_entry {
        key: "XCODE_VERSION_OVERRIDE"
        value: "%{xcode_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_VERSION_OVERRIDE"
        value: "%{apple_sdk_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_PLATFORM"
        value: "%{apple_sdk_platform_value}"
      }
    }
  }
  feature {
    name: "user_link_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "%{user_link_flags}"
        iterate_over: "user_link_flags"
        expand_if_all_available: "user_link_flags"
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
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-headerpad_max_install_names"
        flag: "-no-canonical-prefixes"
        flag: "-target"
        flag: "i386-apple-watchos"
      }
    }
    enabled: true
  }
  feature {
    name: "version_min"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-mwatchos-simulator-version-min=%{version_min}"
      }
    }
  }
  feature {
    name: "dead_strip"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-dead_strip"
        flag: "-no_dead_strip_inits_and_terms"
      }
    }
    requires {
      feature: "opt"
    }
  }
  feature {
    name: "cpp_linker_flags"
  }
  feature {
    name: "apply_implicit_frameworks"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-framework"
        flag: "Foundation"
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "link_cocoa"
  }
  feature {
    name: "apply_simulator_compiler_flags"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fexceptions"
        flag: "-fasm-blocks"
        flag: "-fobjc-abi-version=2"
        flag: "-fobjc-legacy-dispatch"
      }
    }
  }
  feature {
    name: "unfiltered_cxx_flags"
  }
  feature {
    name: "user_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "%{user_compile_flags}"
        iterate_over: "user_compile_flags"
        expand_if_all_available: "user_compile_flags"
      }
    }
  }
  feature {
    name: "unfiltered_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      flag_group {
        flag: "-no-canonical-prefixes"
        flag: "-Wno-builtin-macro-redefined"
        flag: "-D__DATE__=\"redacted\""
        flag: "-D__TIMESTAMP__=\"redacted\""
        flag: "-D__TIME__=\"redacted\""
        flag: "-target"
        flag: "i386-apple-watchos"
      }
    }
  }
  feature {
    name: "linker_param_file"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
      }
    }
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
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
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-c"
        flag: "%{source_file}"
        expand_if_all_available: "source_file"
      }
    }
  }
  feature {
    name: "compiler_output_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-S"
        expand_if_all_available: "output_assembly_file"
      }
      flag_group {
        flag: "-E"
        expand_if_all_available: "output_preprocess_file"
      }
      flag_group {
        flag: "-o"
        flag: "%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "objcopy_embed_flags"
    flag_set {
      action: "objcopy_embed_data"
      flag_group {
        flag: "-I"
        flag: "binary"
      }
    }
    enabled: true
  }
  action_config {
    config_name: "strip"
    action_name: "strip"
    tool {
      tool_path: "/usr/bin/strip"
    }
    flag_set {
      flag_group {
        flag: "-S"
        flag: "-o"
        flag: "%{output_file}"
      }
      flag_group {
        flag: "%{stripopts}"
        iterate_over: "stripopts"
      }
      flag_group {
        flag: "%{input_file}"
      }
    }
  }
  action_config {
    config_name: "c-compile"
    action_name: "c-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-compile"
    action_name: "c++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "linkstamp-compile"
    action_name: "linkstamp-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-module-compile"
    action_name: "c++-module-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-header-parsing"
    action_name: "c++-header-parsing"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-compile"
    action_name: "objc-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "i386"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "objc_actions"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "apply_simulator_compiler_flags"
  }
  action_config {
    config_name: "objc++-compile"
    action_name: "objc++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "i386"
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "apply_simulator_compiler_flags"
  }
  action_config {
    config_name: "assemble"
    action_name: "assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "include_system_dirs"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "preprocess-assemble"
    action_name: "preprocess-assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-archive"
    action_name: "objc-archive"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-filelist"
        flag: "%{obj_list_path}"
        flag: "-arch_only"
        flag: "i386"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{archive_path}"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-executable"
    action_name: "objc-executable"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "i386"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "objc++-executable"
    action_name: "objc++-executable"
    tool {
      tool_path: "wrapped_clang_pp"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
      flag_group {
        flag: "-arch"
        flag: "i386"
      }
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "c++-link-executable"
    action_name: "c++-link-executable"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "symbol_counts"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "force_pic_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-dynamic-library"
    action_name: "c++-link-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-nodeps-dynamic-library"
    action_name: "c++-link-nodeps-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-static-library"
    action_name: "c++-link-static-library"
    tool {
      tool_path: "wrapped_ar"
      execution_requirement: "requires-darwin"
    }
    implies: "runtime_root_flags"
    implies: "archiver_flags"
    implies: "input_param_flags"
    implies: "linker_param_file"
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-fully-link"
    action_name: "objc-fully-link"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-arch_only"
        flag: "i386"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{fully_linked_archive_path}"
      }
      flag_group {
        flag: "%{objc_library_exec_paths}"
        iterate_over: "objc_library_exec_paths"
      }
      flag_group {
        flag: "%{cc_library_exec_paths}"
        iterate_over: "cc_library_exec_paths"
      }
      flag_group {
        flag: "%{imported_library_exec_paths}"
        iterate_over: "imported_library_exec_paths"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objcopy_embed_data"
    action_name: "objcopy_embed_data"
    tool {
      tool_path: "/usr/bin/objcopy"
    }
    enabled: true
  }
  cc_target_os: "apple"
}
toolchain {
  toolchain_identifier: "watchos_x86_64"
  host_system_name: "x86_64-apple-macosx"
  target_system_name: "x86_64-apple-watchos"
  target_cpu: "watchos_x86_64"
  target_libc: "watchos"
  compiler: "compiler"
  abi_version: "local"
  abi_libc_version: "local"
  tool_path {
    name: "ar"
    path: "wrapped_ar"
  }
  tool_path {
    name: "compat-ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "cpp"
    path: "/usr/bin/cpp"
  }
  tool_path {
    name: "dwp"
    path: "/usr/bin/dwp"
  }
  tool_path {
    name: "gcc"
    path: "cc_wrapper.sh"
  }
  tool_path {
    name: "gcov"
    path: "/usr/bin/gcov"
  }
  tool_path {
    name: "ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "nm"
    path: "/usr/bin/nm"
  }
  tool_path {
    name: "objcopy"
    path: "/usr/bin/objcopy"
  }
  tool_path {
    name: "objdump"
    path: "/usr/bin/objdump"
  }
  tool_path {
    name: "strip"
    path: "/usr/bin/strip"
  }
  make_variable {
    name: "STACK_FRAME_UNLIMITED"
    value: "-Wframe-larger-than=100000000 -Wno-vla"
  }
  %{cxx_builtin_include_directory}
  feature {
    name: "fastbuild"
  }
  feature {
    name: "no_legacy_features"
  }
  feature {
    name: "opt"
  }
  feature {
    name: "dbg"
  }
  feature {
    name: "link_libc++"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-lc++"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    enabled: true
  }
  feature {
    name: "compile_all_modules"
  }
  feature {
    name: "exclude_private_headers_in_module_maps"
  }
  feature {
    name: "has_configured_linker_path"
  }
  feature {
    name: "only_doth_headers_in_module_maps"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-fstack-protector"
        flag: "-fcolor-diagnostics"
        flag: "-Wall"
        flag: "-Wthread-safety"
        flag: "-Wself-assign"
        flag: "-fno-omit-frame-pointer"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-O0"
        flag: "-DDEBUG"
      }
      with_feature {
        feature: "fastbuild"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g0"
        flag: "-O2"
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-DNDEBUG"
        flag: "-ffunction-sections"
        flag: "-fdata-sections"
        flag: "-DNS_BLOCK_ASSERTIONS=1"
      }
      with_feature {
        feature: "opt"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g"
      }
      with_feature {
        feature: "dbg"
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
        flag: "-std=c++11"
      }
    }
    enabled: true
  }
  feature {
    name: "debug_prefix_map_pwd_is_dot"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "DEBUG_PREFIX_MAP_PWD=."
      }
    }
  }
  feature {
    name: "generate_dsym_file"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-g"
      }
    }
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "DSYM_HINT_LINKED_BINARY=%{linked_binary}"
        flag: "DSYM_HINT_DSYM_PATH=%{dsym_path}"
      }
    }
  }
  feature {
    name: "contains_objc_source"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fobjc-link-runtime"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "objc_actions"
    implies: "objc-compile"
    implies: "objc++-compile"
    implies: "objc-fully-link"
    implies: "objc-archive"
    implies: "objc-executable"
    implies: "objc++-executable"
    implies: "assemble"
    implies: "preprocess-assemble"
    implies: "c-compile"
    implies: "c++-compile"
    implies: "c++-link-static-library"
    implies: "c++-link-dynamic-library"
    implies: "c++-link-nodeps-dynamic-library"
    implies: "c++-link-executable"
  }
  feature {
    name: "strip_debug_symbols"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-Wl,-S"
        expand_if_all_available: "strip_debug_symbols"
      }
    }
  }
  feature {
    name: "symbol_counts"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,--print-symbol-counts=%{symbol_counts_output}"
        expand_if_all_available: "symbol_counts_output"
      }
    }
  }
  feature {
    name: "shared_flag"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-shared"
      }
    }
  }
  feature {
    name: "kernel_extension"
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
        flag: "-o"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "archiver_flags"
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "rcS"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "runtime_root_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-Wl,-rpath,@loader_path/%{runtime_library_search_directories}"
        iterate_over: "runtime_library_search_directories"
        expand_if_all_available: "runtime_library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_flags}"
        iterate_over: "runtime_root_flags"
        expand_if_all_available: "runtime_root_flags"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_entries}"
        iterate_over: "runtime_root_entries"
        expand_if_all_available: "runtime_root_entries"
      }
    }
  }
  feature {
    name: "input_param_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-L%{library_search_directories}"
        iterate_over: "library_search_directories"
        expand_if_all_available: "library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
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
        flag: "-Wl,-force_load,%{whole_archive_linker_params}"
        iterate_over: "whole_archive_linker_params"
        expand_if_all_available: "whole_archive_linker_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{linker_input_params}"
        iterate_over: "linker_input_params"
        expand_if_all_available: "linker_input_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag_group {
          flag: "-Wl,--start-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.object_files}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.object_files}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          iterate_over: "libraries_to_link.object_files"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag: "-Wl,--end-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
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
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "static_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "dynamic_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l:%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l:%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "versioned_dynamic_library"
          }
        }
        iterate_over: "libraries_to_link"
        expand_if_all_available: "libraries_to_link"
      }
    }
  }
  feature {
    name: "force_pic_flags"
    flag_set {
      action: "c++-link-executable"
      flag_group {
        flag: "-Wl,-pie"
        expand_if_all_available: "force_pic"
      }
    }
  }
  feature {
    name: "pch"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-include"
        flag: "%{pch_file}"
      }
    }
  }
  feature {
    name: "module_maps"
  }
  feature {
    name: "use_objc_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-name=%{module_name}"
        flag: "-iquote"
        flag: "%{module_maps_dir}"
        flag: "-fmodules-cache-path=%{modules_cache_path}"
      }
    }
  }
  feature {
    name: "no_enable_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-maps"
      }
    }
    requires {
      feature: "use_objc_modules"
    }
  }
  feature {
    name: "apply_default_warnings"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-Wshorten-64-to-32"
        flag: "-Wbool-conversion"
        flag: "-Wconstant-conversion"
        flag: "-Wduplicate-method-match"
        flag: "-Wempty-body"
        flag: "-Wenum-conversion"
        flag: "-Wint-conversion"
        flag: "-Wunreachable-code"
        flag: "-Wmismatched-return-types"
        flag: "-Wundeclared-selector"
        flag: "-Wuninitialized"
        flag: "-Wunused-function"
        flag: "-Wunused-variable"
      }
    }
  }
  feature {
    name: "includes"
    flag_set {
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "clif-match"
      flag_group {
        flag: "-include"
        flag: "%{includes}"
        iterate_over: "includes"
        expand_if_all_available: "includes"
      }
    }
    enabled: true
  }
  feature {
    name: "include_paths"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "clif-match"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-iquote"
        flag: "%{quote_include_paths}"
        iterate_over: "quote_include_paths"
      }
      flag_group {
        flag: "-I%{include_paths}"
        iterate_over: "include_paths"
      }
      flag_group {
        flag: "-isystem"
        flag: "%{system_include_paths}"
        iterate_over: "system_include_paths"
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
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "linkstamp-compile"
      action: "clif-match"
      flag_group {
        flag: "--sysroot=%{sysroot}"
        expand_if_all_available: "sysroot"
      }
    }
  }
  feature {
    name: "dependency_file"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      flag_group {
        flag: "-MD"
        flag: "-MF"
        flag: "%{dependency_file}"
        expand_if_all_available: "dependency_file"
      }
    }
  }
  feature {
    name: "pic"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "preprocess-assemble"
      flag_group {
        flag: "-fPIC"
        expand_if_all_available: "pic"
      }
    }
  }
  feature {
    name: "per_object_debug_info"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-gsplit-dwarf"
        expand_if_all_available: "per_object_debug_info_file"
      }
    }
  }
  feature {
    name: "preprocessor_defines"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D%{preprocessor_defines}"
        iterate_over: "preprocessor_defines"
      }
    }
  }
  feature {
    name: "framework_paths"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-F%{framework_paths}"
        iterate_over: "framework_paths"
      }
    }
  }
  feature {
    name: "random_seed"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      flag_group {
        flag: "-frandom-seed=%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "fdo_instrument"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fprofile-generate=%{fdo_instrument_path}"
        flag: "-fno-data-sections"
        expand_if_all_available: "fdo_instrument_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "fdo_optimize"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fprofile-use=%{fdo_profile_path}"
        flag: "-Xclang-only=-Wno-profile-instr-unprofiled"
        flag: "-Xclang-only=-Wno-profile-instr-out-of-date"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "autofdo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fauto-profile=%{fdo_profile_path}"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "lipo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fripa"
      }
    }
    requires {
      feature: "autofdo"
    }
    requires {
      feature: "fdo_optimize"
    }
    requires {
      feature: "fdo_instrument"
    }
  }
  feature {
    name: "coverage"
  }
  feature {
    name: "llvm_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-instr-generate"
        flag: "-fcoverage-mapping"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-fprofile-instr-generate"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "gcc_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-arcs"
        flag: "-ftest-coverage"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "--coverage"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "apply_default_compiler_flags"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-DOS_IOS"
        flag: "-fno-autolink"
      }
    }
  }
  feature {
    name: "include_system_dirs"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-isysroot"
        flag: "%{sdk_dir}"
        flag: "-F%{sdk_framework_dir}"
        flag: "-F%{platform_developer_framework_dir}"
      }
    }
  }
  feature {
    name: "bitcode_embedded"
  }
  feature {
    name: "bitcode_embedded_markers"
  }
  feature {
    name: "objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fobjc-arc"
        expand_if_all_available: "objc_arc"
      }
    }
  }
  feature {
    name: "no_objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fno-objc-arc"
        expand_if_all_available: "no_objc_arc"
      }
    }
  }
  feature {
    name: "apple_env"
    env_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-archive"
      action: "objc-fully-link"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      action: "objc-executable"
      action: "objc++-executable"
      action: "linkstamp-compile"
      env_entry {
        key: "XCODE_VERSION_OVERRIDE"
        value: "%{xcode_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_VERSION_OVERRIDE"
        value: "%{apple_sdk_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_PLATFORM"
        value: "%{apple_sdk_platform_value}"
      }
    }
  }
  feature {
    name: "user_link_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "%{user_link_flags}"
        iterate_over: "user_link_flags"
        expand_if_all_available: "user_link_flags"
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
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-headerpad_max_install_names"
        flag: "-no-canonical-prefixes"
      }
    }
    enabled: true
  }
  feature {
    name: "version_min"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-m<platform_for_version_min>-version-min=%{version_min}"
      }
    }
  }
  feature {
    name: "dead_strip"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-dead_strip"
        flag: "-no_dead_strip_inits_and_terms"
      }
    }
    requires {
      feature: "opt"
    }
  }
  feature {
    name: "cpp_linker_flags"
  }
  feature {
    name: "apply_implicit_frameworks"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-framework"
        flag: "Foundation"
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "link_cocoa"
  }
  feature {
    name: "apply_simulator_compiler_flags"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fexceptions"
        flag: "-fasm-blocks"
        flag: "-fobjc-abi-version=2"
        flag: "-fobjc-legacy-dispatch"
      }
    }
  }
  feature {
    name: "unfiltered_cxx_flags"
  }
  feature {
    name: "user_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "%{user_compile_flags}"
        iterate_over: "user_compile_flags"
        expand_if_all_available: "user_compile_flags"
      }
    }
  }
  feature {
    name: "unfiltered_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      flag_group {
        flag: "-no-canonical-prefixes"
        flag: "-Wno-builtin-macro-redefined"
        flag: "-D__DATE__=\"redacted\""
        flag: "-D__TIMESTAMP__=\"redacted\""
        flag: "-D__TIME__=\"redacted\""
      }
    }
  }
  feature {
    name: "linker_param_file"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
      }
    }
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
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
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-c"
        flag: "%{source_file}"
        expand_if_all_available: "source_file"
      }
    }
  }
  feature {
    name: "compiler_output_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-S"
        expand_if_all_available: "output_assembly_file"
      }
      flag_group {
        flag: "-E"
        expand_if_all_available: "output_preprocess_file"
      }
      flag_group {
        flag: "-o"
        flag: "%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "objcopy_embed_flags"
    flag_set {
      action: "objcopy_embed_data"
      flag_group {
        flag: "-I"
        flag: "binary"
      }
    }
    enabled: true
  }
  action_config {
    config_name: "strip"
    action_name: "strip"
    tool {
      tool_path: "/usr/bin/strip"
    }
    flag_set {
      flag_group {
        flag: "-S"
        flag: "-o"
        flag: "%{output_file}"
      }
      flag_group {
        flag: "%{stripopts}"
        iterate_over: "stripopts"
      }
      flag_group {
        flag: "%{input_file}"
      }
    }
  }
  action_config {
    config_name: "c-compile"
    action_name: "c-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-compile"
    action_name: "c++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "linkstamp-compile"
    action_name: "linkstamp-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-module-compile"
    action_name: "c++-module-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-header-parsing"
    action_name: "c++-header-parsing"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-compile"
    action_name: "objc-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "x86_64"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "objc_actions"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "apply_simulator_compiler_flags"
  }
  action_config {
    config_name: "objc++-compile"
    action_name: "objc++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "x86_64"
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "apply_simulator_compiler_flags"
  }
  action_config {
    config_name: "assemble"
    action_name: "assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "include_system_dirs"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "preprocess-assemble"
    action_name: "preprocess-assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-archive"
    action_name: "objc-archive"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-filelist"
        flag: "%{obj_list_path}"
        flag: "-arch_only"
        flag: "x86_64"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{archive_path}"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-executable"
    action_name: "objc-executable"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "x86_64"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "objc++-executable"
    action_name: "objc++-executable"
    tool {
      tool_path: "wrapped_clang_pp"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
      flag_group {
        flag: "-arch"
        flag: "x86_64"
      }
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "c++-link-executable"
    action_name: "c++-link-executable"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "symbol_counts"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "force_pic_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-dynamic-library"
    action_name: "c++-link-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-nodeps-dynamic-library"
    action_name: "c++-link-nodeps-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-static-library"
    action_name: "c++-link-static-library"
    tool {
      tool_path: "wrapped_ar"
      execution_requirement: "requires-darwin"
    }
    implies: "runtime_root_flags"
    implies: "archiver_flags"
    implies: "input_param_flags"
    implies: "linker_param_file"
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-fully-link"
    action_name: "objc-fully-link"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-arch_only"
        flag: "x86_64"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{fully_linked_archive_path}"
      }
      flag_group {
        flag: "%{objc_library_exec_paths}"
        iterate_over: "objc_library_exec_paths"
      }
      flag_group {
        flag: "%{cc_library_exec_paths}"
        iterate_over: "cc_library_exec_paths"
      }
      flag_group {
        flag: "%{imported_library_exec_paths}"
        iterate_over: "imported_library_exec_paths"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objcopy_embed_data"
    action_name: "objcopy_embed_data"
    tool {
      tool_path: "/usr/bin/objcopy"
    }
    enabled: true
  }
  cc_target_os: "apple"
}
toolchain {
  toolchain_identifier: "watchos_arm64_32"
  host_system_name: "x86_64-apple-macosx"
  target_system_name: "arm64_32-apple-watchos"
  target_cpu: "watchos_arm64_32"
  target_libc: "watchos"
  compiler: "compiler"
  abi_version: "local"
  abi_libc_version: "local"
  tool_path {
    name: "ar"
    path: "wrapped_ar"
  }
  tool_path {
    name: "compat-ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "cpp"
    path: "/usr/bin/cpp"
  }
  tool_path {
    name: "dwp"
    path: "/usr/bin/dwp"
  }
  tool_path {
    name: "gcc"
    path: "cc_wrapper.sh"
  }
  tool_path {
    name: "gcov"
    path: "/usr/bin/gcov"
  }
  tool_path {
    name: "ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "nm"
    path: "/usr/bin/nm"
  }
  tool_path {
    name: "objcopy"
    path: "/usr/bin/objcopy"
  }
  tool_path {
    name: "objdump"
    path: "/usr/bin/objdump"
  }
  tool_path {
    name: "strip"
    path: "/usr/bin/strip"
  }
  make_variable {
    name: "STACK_FRAME_UNLIMITED"
    value: "-Wframe-larger-than=100000000 -Wno-vla"
  }
  %{cxx_builtin_include_directory}
  feature {
    name: "fastbuild"
  }
  feature {
    name: "no_legacy_features"
  }
  feature {
    name: "opt"
  }
  feature {
    name: "dbg"
  }
  feature {
    name: "link_libc++"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-lc++"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    enabled: true
  }
  feature {
    name: "compile_all_modules"
  }
  feature {
    name: "exclude_private_headers_in_module_maps"
  }
  feature {
    name: "has_configured_linker_path"
  }
  feature {
    name: "only_doth_headers_in_module_maps"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-fstack-protector"
        flag: "-fcolor-diagnostics"
        flag: "-Wall"
        flag: "-Wthread-safety"
        flag: "-Wself-assign"
        flag: "-fno-omit-frame-pointer"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-O0"
        flag: "-DDEBUG"
      }
      with_feature {
        feature: "fastbuild"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g0"
        flag: "-O2"
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-DNDEBUG"
        flag: "-ffunction-sections"
        flag: "-fdata-sections"
        flag: "-DNS_BLOCK_ASSERTIONS=1"
      }
      with_feature {
        feature: "opt"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g"
      }
      with_feature {
        feature: "dbg"
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
        flag: "-std=c++11"
      }
    }
    enabled: true
  }
  feature {
    name: "debug_prefix_map_pwd_is_dot"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "DEBUG_PREFIX_MAP_PWD=."
      }
    }
  }
  feature {
    name: "generate_dsym_file"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-g"
      }
    }
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "DSYM_HINT_LINKED_BINARY=%{linked_binary}"
        flag: "DSYM_HINT_DSYM_PATH=%{dsym_path}"
      }
    }
  }
  feature {
    name: "contains_objc_source"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fobjc-link-runtime"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "objc_actions"
    implies: "objc-compile"
    implies: "objc++-compile"
    implies: "objc-fully-link"
    implies: "objc-archive"
    implies: "objc-executable"
    implies: "objc++-executable"
    implies: "assemble"
    implies: "preprocess-assemble"
    implies: "c-compile"
    implies: "c++-compile"
    implies: "c++-link-static-library"
    implies: "c++-link-dynamic-library"
    implies: "c++-link-nodeps-dynamic-library"
    implies: "c++-link-executable"
  }
  feature {
    name: "strip_debug_symbols"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-Wl,-S"
        expand_if_all_available: "strip_debug_symbols"
      }
    }
  }
  feature {
    name: "symbol_counts"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,--print-symbol-counts=%{symbol_counts_output}"
        expand_if_all_available: "symbol_counts_output"
      }
    }
  }
  feature {
    name: "shared_flag"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-shared"
      }
    }
  }
  feature {
    name: "kernel_extension"
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
        flag: "-o"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "archiver_flags"
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "rcS"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "runtime_root_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-Wl,-rpath,@loader_path/%{runtime_library_search_directories}"
        iterate_over: "runtime_library_search_directories"
        expand_if_all_available: "runtime_library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_flags}"
        iterate_over: "runtime_root_flags"
        expand_if_all_available: "runtime_root_flags"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_entries}"
        iterate_over: "runtime_root_entries"
        expand_if_all_available: "runtime_root_entries"
      }
    }
  }
  feature {
    name: "input_param_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-L%{library_search_directories}"
        iterate_over: "library_search_directories"
        expand_if_all_available: "library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
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
        flag: "-Wl,-force_load,%{whole_archive_linker_params}"
        iterate_over: "whole_archive_linker_params"
        expand_if_all_available: "whole_archive_linker_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{linker_input_params}"
        iterate_over: "linker_input_params"
        expand_if_all_available: "linker_input_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag_group {
          flag: "-Wl,--start-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.object_files}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.object_files}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          iterate_over: "libraries_to_link.object_files"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag: "-Wl,--end-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
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
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "static_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "dynamic_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l:%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l:%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "versioned_dynamic_library"
          }
        }
        iterate_over: "libraries_to_link"
        expand_if_all_available: "libraries_to_link"
      }
    }
  }
  feature {
    name: "force_pic_flags"
    flag_set {
      action: "c++-link-executable"
      flag_group {
        flag: "-Wl,-pie"
        expand_if_all_available: "force_pic"
      }
    }
  }
  feature {
    name: "pch"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-include"
        flag: "%{pch_file}"
      }
    }
  }
  feature {
    name: "module_maps"
  }
  feature {
    name: "use_objc_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-name=%{module_name}"
        flag: "-iquote"
        flag: "%{module_maps_dir}"
        flag: "-fmodules-cache-path=%{modules_cache_path}"
      }
    }
  }
  feature {
    name: "no_enable_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-maps"
      }
    }
    requires {
      feature: "use_objc_modules"
    }
  }
  feature {
    name: "apply_default_warnings"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-Wshorten-64-to-32"
        flag: "-Wbool-conversion"
        flag: "-Wconstant-conversion"
        flag: "-Wduplicate-method-match"
        flag: "-Wempty-body"
        flag: "-Wenum-conversion"
        flag: "-Wint-conversion"
        flag: "-Wunreachable-code"
        flag: "-Wmismatched-return-types"
        flag: "-Wundeclared-selector"
        flag: "-Wuninitialized"
        flag: "-Wunused-function"
        flag: "-Wunused-variable"
      }
    }
  }
  feature {
    name: "includes"
    flag_set {
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "clif-match"
      flag_group {
        flag: "-include"
        flag: "%{includes}"
        iterate_over: "includes"
        expand_if_all_available: "includes"
      }
    }
    enabled: true
  }
  feature {
    name: "include_paths"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "clif-match"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-iquote"
        flag: "%{quote_include_paths}"
        iterate_over: "quote_include_paths"
      }
      flag_group {
        flag: "-I%{include_paths}"
        iterate_over: "include_paths"
      }
      flag_group {
        flag: "-isystem"
        flag: "%{system_include_paths}"
        iterate_over: "system_include_paths"
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
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "linkstamp-compile"
      action: "clif-match"
      flag_group {
        flag: "--sysroot=%{sysroot}"
        expand_if_all_available: "sysroot"
      }
    }
  }
  feature {
    name: "dependency_file"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      flag_group {
        flag: "-MD"
        flag: "-MF"
        flag: "%{dependency_file}"
        expand_if_all_available: "dependency_file"
      }
    }
  }
  feature {
    name: "pic"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "preprocess-assemble"
      flag_group {
        flag: "-fPIC"
        expand_if_all_available: "pic"
      }
    }
  }
  feature {
    name: "per_object_debug_info"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-gsplit-dwarf"
        expand_if_all_available: "per_object_debug_info_file"
      }
    }
  }
  feature {
    name: "preprocessor_defines"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D%{preprocessor_defines}"
        iterate_over: "preprocessor_defines"
      }
    }
  }
  feature {
    name: "framework_paths"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-F%{framework_paths}"
        iterate_over: "framework_paths"
      }
    }
  }
  feature {
    name: "random_seed"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      flag_group {
        flag: "-frandom-seed=%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "fdo_instrument"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fprofile-generate=%{fdo_instrument_path}"
        flag: "-fno-data-sections"
        expand_if_all_available: "fdo_instrument_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "fdo_optimize"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fprofile-use=%{fdo_profile_path}"
        flag: "-Xclang-only=-Wno-profile-instr-unprofiled"
        flag: "-Xclang-only=-Wno-profile-instr-out-of-date"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "autofdo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fauto-profile=%{fdo_profile_path}"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "lipo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fripa"
      }
    }
    requires {
      feature: "autofdo"
    }
    requires {
      feature: "fdo_optimize"
    }
    requires {
      feature: "fdo_instrument"
    }
  }
  feature {
    name: "coverage"
  }
  feature {
    name: "llvm_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-instr-generate"
        flag: "-fcoverage-mapping"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-fprofile-instr-generate"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "gcc_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-arcs"
        flag: "-ftest-coverage"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "--coverage"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "apply_default_compiler_flags"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-DOS_IOS"
        flag: "-fno-autolink"
      }
    }
  }
  feature {
    name: "include_system_dirs"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-isysroot"
        flag: "%{sdk_dir}"
        flag: "-F%{sdk_framework_dir}"
        flag: "-F%{platform_developer_framework_dir}"
      }
    }
  }
  feature {
    name: "bitcode_embedded"
  }
  feature {
    name: "bitcode_embedded_markers"
  }
  feature {
    name: "objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fobjc-arc"
        expand_if_all_available: "objc_arc"
      }
    }
  }
  feature {
    name: "no_objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fno-objc-arc"
        expand_if_all_available: "no_objc_arc"
      }
    }
  }
  feature {
    name: "apple_env"
    env_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-archive"
      action: "objc-fully-link"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      action: "objc-executable"
      action: "objc++-executable"
      action: "linkstamp-compile"
      env_entry {
        key: "XCODE_VERSION_OVERRIDE"
        value: "%{xcode_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_VERSION_OVERRIDE"
        value: "%{apple_sdk_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_PLATFORM"
        value: "%{apple_sdk_platform_value}"
      }
    }
  }
  feature {
    name: "user_link_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "%{user_link_flags}"
        iterate_over: "user_link_flags"
        expand_if_all_available: "user_link_flags"
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
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-headerpad_max_install_names"
        flag: "-no-canonical-prefixes"
      }
    }
    enabled: true
  }
  feature {
    name: "version_min"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-m<platform_for_version_min>-version-min=%{version_min}"
      }
    }
  }
  feature {
    name: "dead_strip"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-dead_strip"
        flag: "-no_dead_strip_inits_and_terms"
      }
    }
    requires {
      feature: "opt"
    }
  }
  feature {
    name: "cpp_linker_flags"
  }
  feature {
    name: "apply_implicit_frameworks"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-framework"
        flag: "Foundation"
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "link_cocoa"
  }
  feature {
    name: "apply_simulator_compiler_flags"
  }
  feature {
    name: "unfiltered_cxx_flags"
  }
  feature {
    name: "user_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "%{user_compile_flags}"
        iterate_over: "user_compile_flags"
        expand_if_all_available: "user_compile_flags"
      }
    }
  }
  feature {
    name: "unfiltered_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      flag_group {
        flag: "-no-canonical-prefixes"
        flag: "-Wno-builtin-macro-redefined"
        flag: "-D__DATE__=\"redacted\""
        flag: "-D__TIMESTAMP__=\"redacted\""
        flag: "-D__TIME__=\"redacted\""
      }
    }
  }
  feature {
    name: "linker_param_file"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
      }
    }
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
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
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-c"
        flag: "%{source_file}"
        expand_if_all_available: "source_file"
      }
    }
  }
  feature {
    name: "compiler_output_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-S"
        expand_if_all_available: "output_assembly_file"
      }
      flag_group {
        flag: "-E"
        expand_if_all_available: "output_preprocess_file"
      }
      flag_group {
        flag: "-o"
        flag: "%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "objcopy_embed_flags"
    flag_set {
      action: "objcopy_embed_data"
      flag_group {
        flag: "-I"
        flag: "binary"
      }
    }
    enabled: true
  }
  action_config {
    config_name: "strip"
    action_name: "strip"
    tool {
      tool_path: "/usr/bin/strip"
    }
    flag_set {
      flag_group {
        flag: "-S"
        flag: "-o"
        flag: "%{output_file}"
      }
      flag_group {
        flag: "%{stripopts}"
        iterate_over: "stripopts"
      }
      flag_group {
        flag: "%{input_file}"
      }
    }
  }
  action_config {
    config_name: "c-compile"
    action_name: "c-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-compile"
    action_name: "c++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "linkstamp-compile"
    action_name: "linkstamp-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-module-compile"
    action_name: "c++-module-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-header-parsing"
    action_name: "c++-header-parsing"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-compile"
    action_name: "objc-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "<architecture>"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "objc_actions"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
  }
  action_config {
    config_name: "objc++-compile"
    action_name: "objc++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "<architecture>"
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
  }
  action_config {
    config_name: "assemble"
    action_name: "assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "include_system_dirs"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "preprocess-assemble"
    action_name: "preprocess-assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-archive"
    action_name: "objc-archive"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-filelist"
        flag: "%{obj_list_path}"
        flag: "-arch_only"
        flag: "<architecture>"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{archive_path}"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-executable"
    action_name: "objc-executable"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "<architecture>"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "objc++-executable"
    action_name: "objc++-executable"
    tool {
      tool_path: "wrapped_clang_pp"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
      flag_group {
        flag: "-arch"
        flag: "<architecture>"
      }
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "c++-link-executable"
    action_name: "c++-link-executable"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "symbol_counts"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "force_pic_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-dynamic-library"
    action_name: "c++-link-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-nodeps-dynamic-library"
    action_name: "c++-link-nodeps-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-static-library"
    action_name: "c++-link-static-library"
    tool {
      tool_path: "wrapped_ar"
      execution_requirement: "requires-darwin"
    }
    implies: "runtime_root_flags"
    implies: "archiver_flags"
    implies: "input_param_flags"
    implies: "linker_param_file"
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-fully-link"
    action_name: "objc-fully-link"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-arch_only"
        flag: "<architecture>"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{fully_linked_archive_path}"
      }
      flag_group {
        flag: "%{objc_library_exec_paths}"
        iterate_over: "objc_library_exec_paths"
      }
      flag_group {
        flag: "%{cc_library_exec_paths}"
        iterate_over: "cc_library_exec_paths"
      }
      flag_group {
        flag: "%{imported_library_exec_paths}"
        iterate_over: "imported_library_exec_paths"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objcopy_embed_data"
    action_name: "objcopy_embed_data"
    tool {
      tool_path: "/usr/bin/objcopy"
    }
    enabled: true
  }
  cc_target_os: "apple"
}
toolchain {
  toolchain_identifier: "tvos_x86_64"
  host_system_name: "x86_64-apple-macosx"
  target_system_name: "x86_64-apple-tvos"
  target_cpu: "tvos_x86_64"
  target_libc: "tvos"
  compiler: "compiler"
  abi_version: "local"
  abi_libc_version: "local"
  tool_path {
    name: "ar"
    path: "wrapped_ar"
  }
  tool_path {
    name: "compat-ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "cpp"
    path: "/usr/bin/cpp"
  }
  tool_path {
    name: "dwp"
    path: "/usr/bin/dwp"
  }
  tool_path {
    name: "gcc"
    path: "cc_wrapper.sh"
  }
  tool_path {
    name: "gcov"
    path: "/usr/bin/gcov"
  }
  tool_path {
    name: "ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "nm"
    path: "/usr/bin/nm"
  }
  tool_path {
    name: "objcopy"
    path: "/usr/bin/objcopy"
  }
  tool_path {
    name: "objdump"
    path: "/usr/bin/objdump"
  }
  tool_path {
    name: "strip"
    path: "/usr/bin/strip"
  }
  make_variable {
    name: "STACK_FRAME_UNLIMITED"
    value: "-Wframe-larger-than=100000000 -Wno-vla"
  }
  %{cxx_builtin_include_directory}
  builtin_sysroot: ""
  feature {
    name: "fastbuild"
  }
  feature {
    name: "no_legacy_features"
  }
  feature {
    name: "opt"
  }
  feature {
    name: "dbg"
  }
  feature {
    name: "link_libc++"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-lc++"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    enabled: true
  }
  feature {
    name: "compile_all_modules"
  }
  feature {
    name: "exclude_private_headers_in_module_maps"
  }
  feature {
    name: "has_configured_linker_path"
  }
  feature {
    name: "only_doth_headers_in_module_maps"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-fstack-protector"
        flag: "-fcolor-diagnostics"
        flag: "-Wall"
        flag: "-Wthread-safety"
        flag: "-Wself-assign"
        flag: "-fno-omit-frame-pointer"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-O0"
        flag: "-DDEBUG"
      }
      with_feature {
        feature: "fastbuild"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g0"
        flag: "-O2"
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-DNDEBUG"
        flag: "-ffunction-sections"
        flag: "-fdata-sections"
        flag: "-DNS_BLOCK_ASSERTIONS=1"
      }
      with_feature {
        feature: "opt"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g"
      }
      with_feature {
        feature: "dbg"
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
        flag: "-std=c++11"
      }
    }
    enabled: true
  }
  feature {
    name: "debug_prefix_map_pwd_is_dot"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "DEBUG_PREFIX_MAP_PWD=."
      }
    }
  }
  feature {
    name: "generate_dsym_file"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-g"
      }
    }
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "DSYM_HINT_LINKED_BINARY=%{linked_binary}"
        flag: "DSYM_HINT_DSYM_PATH=%{dsym_path}"
      }
    }
  }
  feature {
    name: "contains_objc_source"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fobjc-link-runtime"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "objc_actions"
    implies: "objc-compile"
    implies: "objc++-compile"
    implies: "objc-fully-link"
    implies: "objc-archive"
    implies: "objc-executable"
    implies: "objc++-executable"
    implies: "assemble"
    implies: "preprocess-assemble"
    implies: "c-compile"
    implies: "c++-compile"
    implies: "c++-link-static-library"
    implies: "c++-link-dynamic-library"
    implies: "c++-link-nodeps-dynamic-library"
    implies: "c++-link-executable"
  }
  feature {
    name: "strip_debug_symbols"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-Wl,-S"
        expand_if_all_available: "strip_debug_symbols"
      }
    }
  }
  feature {
    name: "symbol_counts"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,--print-symbol-counts=%{symbol_counts_output}"
        expand_if_all_available: "symbol_counts_output"
      }
    }
  }
  feature {
    name: "shared_flag"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-shared"
      }
    }
  }
  feature {
    name: "kernel_extension"
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
        flag: "-o"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "archiver_flags"
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "rcS"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "runtime_root_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-Wl,-rpath,@loader_path/%{runtime_library_search_directories}"
        iterate_over: "runtime_library_search_directories"
        expand_if_all_available: "runtime_library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_flags}"
        iterate_over: "runtime_root_flags"
        expand_if_all_available: "runtime_root_flags"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_entries}"
        iterate_over: "runtime_root_entries"
        expand_if_all_available: "runtime_root_entries"
      }
    }
  }
  feature {
    name: "input_param_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-L%{library_search_directories}"
        iterate_over: "library_search_directories"
        expand_if_all_available: "library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
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
        flag: "-Wl,-force_load,%{whole_archive_linker_params}"
        iterate_over: "whole_archive_linker_params"
        expand_if_all_available: "whole_archive_linker_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{linker_input_params}"
        iterate_over: "linker_input_params"
        expand_if_all_available: "linker_input_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag_group {
          flag: "-Wl,--start-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.object_files}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.object_files}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          iterate_over: "libraries_to_link.object_files"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag: "-Wl,--end-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
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
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "static_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "dynamic_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l:%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l:%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "versioned_dynamic_library"
          }
        }
        iterate_over: "libraries_to_link"
        expand_if_all_available: "libraries_to_link"
      }
    }
  }
  feature {
    name: "force_pic_flags"
    flag_set {
      action: "c++-link-executable"
      flag_group {
        flag: "-Wl,-pie"
        expand_if_all_available: "force_pic"
      }
    }
  }
  feature {
    name: "pch"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-include"
        flag: "%{pch_file}"
      }
    }
  }
  feature {
    name: "module_maps"
  }
  feature {
    name: "use_objc_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-name=%{module_name}"
        flag: "-iquote"
        flag: "%{module_maps_dir}"
        flag: "-fmodules-cache-path=%{modules_cache_path}"
      }
    }
  }
  feature {
    name: "no_enable_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-maps"
      }
    }
    requires {
      feature: "use_objc_modules"
    }
  }
  feature {
    name: "apply_default_warnings"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-Wshorten-64-to-32"
        flag: "-Wbool-conversion"
        flag: "-Wconstant-conversion"
        flag: "-Wduplicate-method-match"
        flag: "-Wempty-body"
        flag: "-Wenum-conversion"
        flag: "-Wint-conversion"
        flag: "-Wunreachable-code"
        flag: "-Wmismatched-return-types"
        flag: "-Wundeclared-selector"
        flag: "-Wuninitialized"
        flag: "-Wunused-function"
        flag: "-Wunused-variable"
      }
    }
  }
  feature {
    name: "includes"
    flag_set {
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "clif-match"
      flag_group {
        flag: "-include"
        flag: "%{includes}"
        iterate_over: "includes"
        expand_if_all_available: "includes"
      }
    }
    enabled: true
  }
  feature {
    name: "include_paths"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "clif-match"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-iquote"
        flag: "%{quote_include_paths}"
        iterate_over: "quote_include_paths"
      }
      flag_group {
        flag: "-I%{include_paths}"
        iterate_over: "include_paths"
      }
      flag_group {
        flag: "-isystem"
        flag: "%{system_include_paths}"
        iterate_over: "system_include_paths"
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
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "linkstamp-compile"
      action: "clif-match"
      flag_group {
        flag: "--sysroot=%{sysroot}"
        expand_if_all_available: "sysroot"
      }
    }
  }
  feature {
    name: "dependency_file"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      flag_group {
        flag: "-MD"
        flag: "-MF"
        flag: "%{dependency_file}"
        expand_if_all_available: "dependency_file"
      }
    }
  }
  feature {
    name: "pic"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "preprocess-assemble"
      flag_group {
        flag: "-fPIC"
        expand_if_all_available: "pic"
      }
    }
  }
  feature {
    name: "per_object_debug_info"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-gsplit-dwarf"
        expand_if_all_available: "per_object_debug_info_file"
      }
    }
  }
  feature {
    name: "preprocessor_defines"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D%{preprocessor_defines}"
        iterate_over: "preprocessor_defines"
      }
    }
  }
  feature {
    name: "framework_paths"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-F%{framework_paths}"
        iterate_over: "framework_paths"
      }
    }
  }
  feature {
    name: "random_seed"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      flag_group {
        flag: "-frandom-seed=%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "fdo_instrument"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fprofile-generate=%{fdo_instrument_path}"
        flag: "-fno-data-sections"
        expand_if_all_available: "fdo_instrument_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "fdo_optimize"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fprofile-use=%{fdo_profile_path}"
        flag: "-Xclang-only=-Wno-profile-instr-unprofiled"
        flag: "-Xclang-only=-Wno-profile-instr-out-of-date"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "autofdo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fauto-profile=%{fdo_profile_path}"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "lipo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fripa"
      }
    }
    requires {
      feature: "autofdo"
    }
    requires {
      feature: "fdo_optimize"
    }
    requires {
      feature: "fdo_instrument"
    }
  }
  feature {
    name: "coverage"
  }
  feature {
    name: "llvm_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-instr-generate"
        flag: "-fcoverage-mapping"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-fprofile-instr-generate"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "gcc_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-arcs"
        flag: "-ftest-coverage"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "--coverage"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "apply_default_compiler_flags"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-DOS_TVOS"
        flag: "-fno-autolink"
      }
    }
  }
  feature {
    name: "include_system_dirs"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-isysroot"
        flag: "%{sdk_dir}"
      }
    }
  }
  feature {
    name: "bitcode_embedded"
  }
  feature {
    name: "bitcode_embedded_markers"
  }
  feature {
    name: "objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fobjc-arc"
        expand_if_all_available: "objc_arc"
      }
    }
  }
  feature {
    name: "no_objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fno-objc-arc"
        expand_if_all_available: "no_objc_arc"
      }
    }
  }
  feature {
    name: "apple_env"
    env_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-archive"
      action: "objc-fully-link"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      action: "objc-executable"
      action: "objc++-executable"
      action: "linkstamp-compile"
      env_entry {
        key: "XCODE_VERSION_OVERRIDE"
        value: "%{xcode_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_VERSION_OVERRIDE"
        value: "%{apple_sdk_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_PLATFORM"
        value: "%{apple_sdk_platform_value}"
      }
    }
  }
  feature {
    name: "user_link_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "%{user_link_flags}"
        iterate_over: "user_link_flags"
        expand_if_all_available: "user_link_flags"
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
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-headerpad_max_install_names"
        flag: "-no-canonical-prefixes"
        flag: "-target"
        flag: "x86_64-apple-tvos"
      }
    }
    enabled: true
  }
  feature {
    name: "version_min"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-mtvos-simulator-version-min=%{version_min}"
      }
    }
  }
  feature {
    name: "dead_strip"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-dead_strip"
        flag: "-no_dead_strip_inits_and_terms"
      }
    }
    requires {
      feature: "opt"
    }
  }
  feature {
    name: "cpp_linker_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      flag_group {
        flag: "-lc++"
        flag: "-target"
        flag: "x86_64-apple-tvos"
      }
    }
  }
  feature {
    name: "apply_implicit_frameworks"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-framework"
        flag: "Foundation"
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "link_cocoa"
  }
  feature {
    name: "apply_simulator_compiler_flags"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fexceptions"
        flag: "-fasm-blocks"
        flag: "-fobjc-abi-version=2"
        flag: "-fobjc-legacy-dispatch"
      }
    }
  }
  feature {
    name: "unfiltered_cxx_flags"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-no-canonical-prefixes"
        flag: "-pthread"
      }
    }
  }
  feature {
    name: "user_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "%{user_compile_flags}"
        iterate_over: "user_compile_flags"
        expand_if_all_available: "user_compile_flags"
      }
    }
  }
  feature {
    name: "unfiltered_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      flag_group {
        flag: "-no-canonical-prefixes"
        flag: "-Wno-builtin-macro-redefined"
        flag: "-D__DATE__=\"redacted\""
        flag: "-D__TIMESTAMP__=\"redacted\""
        flag: "-D__TIME__=\"redacted\""
        flag: "-target"
        flag: "x86_64-apple-tvos"
      }
    }
  }
  feature {
    name: "linker_param_file"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
      }
    }
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
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
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-c"
        flag: "%{source_file}"
        expand_if_all_available: "source_file"
      }
    }
  }
  feature {
    name: "compiler_output_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-S"
        expand_if_all_available: "output_assembly_file"
      }
      flag_group {
        flag: "-E"
        expand_if_all_available: "output_preprocess_file"
      }
      flag_group {
        flag: "-o"
        flag: "%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "objcopy_embed_flags"
    flag_set {
      action: "objcopy_embed_data"
      flag_group {
        flag: "-I"
        flag: "binary"
      }
    }
    enabled: true
  }
  action_config {
    config_name: "strip"
    action_name: "strip"
    tool {
      tool_path: "/usr/bin/strip"
    }
    flag_set {
      flag_group {
        flag: "-S"
        flag: "-o"
        flag: "%{output_file}"
      }
      flag_group {
        flag: "%{stripopts}"
        iterate_over: "stripopts"
      }
      flag_group {
        flag: "%{input_file}"
      }
    }
  }
  action_config {
    config_name: "c-compile"
    action_name: "c-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "unfiltered_cxx_flags"
  }
  action_config {
    config_name: "c++-compile"
    action_name: "c++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "unfiltered_cxx_flags"
  }
  action_config {
    config_name: "linkstamp-compile"
    action_name: "linkstamp-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-module-compile"
    action_name: "c++-module-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "unfiltered_cxx_flags"
  }
  action_config {
    config_name: "c++-header-parsing"
    action_name: "c++-header-parsing"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "unfiltered_cxx_flags"
  }
  action_config {
    config_name: "objc-compile"
    action_name: "objc-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "x86_64"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "objc_actions"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "apply_simulator_compiler_flags"
  }
  action_config {
    config_name: "objc++-compile"
    action_name: "objc++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "x86_64"
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "apply_simulator_compiler_flags"
  }
  action_config {
    config_name: "assemble"
    action_name: "assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "include_system_dirs"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "unfiltered_cxx_flags"
  }
  action_config {
    config_name: "preprocess-assemble"
    action_name: "preprocess-assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "unfiltered_cxx_flags"
  }
  action_config {
    config_name: "objc-archive"
    action_name: "objc-archive"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-filelist"
        flag: "%{obj_list_path}"
        flag: "-arch_only"
        flag: "x86_64"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{archive_path}"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-executable"
    action_name: "objc-executable"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "x86_64"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "objc++-executable"
    action_name: "objc++-executable"
    tool {
      tool_path: "wrapped_clang_pp"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
      flag_group {
        flag: "-arch"
        flag: "x86_64"
      }
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "c++-link-executable"
    action_name: "c++-link-executable"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "symbol_counts"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "force_pic_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
    implies: "cpp_linker_flags"
  }
  action_config {
    config_name: "c++-link-dynamic-library"
    action_name: "c++-link-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
    implies: "cpp_linker_flags"
  }
  action_config {
    config_name: "c++-link-nodeps-dynamic-library"
    action_name: "c++-link-nodeps-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
    implies: "cpp_linker_flags"
  }
  action_config {
    config_name: "c++-link-static-library"
    action_name: "c++-link-static-library"
    tool {
      tool_path: "wrapped_ar"
      execution_requirement: "requires-darwin"
    }
    implies: "runtime_root_flags"
    implies: "archiver_flags"
    implies: "input_param_flags"
    implies: "linker_param_file"
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-fully-link"
    action_name: "objc-fully-link"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-arch_only"
        flag: "x86_64"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{fully_linked_archive_path}"
      }
      flag_group {
        flag: "%{objc_library_exec_paths}"
        iterate_over: "objc_library_exec_paths"
      }
      flag_group {
        flag: "%{cc_library_exec_paths}"
        iterate_over: "cc_library_exec_paths"
      }
      flag_group {
        flag: "%{imported_library_exec_paths}"
        iterate_over: "imported_library_exec_paths"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objcopy_embed_data"
    action_name: "objcopy_embed_data"
    tool {
      tool_path: "/usr/bin/objcopy"
    }
    enabled: true
  }
  cc_target_os: "apple"
}
toolchain {
  toolchain_identifier: "ios_i386"
  host_system_name: "x86_64-apple-macosx"
  target_system_name: "i386-apple-ios"
  target_cpu: "ios_i386"
  target_libc: "ios"
  compiler: "compiler"
  abi_version: "local"
  abi_libc_version: "local"
  tool_path {
    name: "ar"
    path: "wrapped_ar"
  }
  tool_path {
    name: "compat-ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "cpp"
    path: "/usr/bin/cpp"
  }
  tool_path {
    name: "dwp"
    path: "/usr/bin/dwp"
  }
  tool_path {
    name: "gcc"
    path: "cc_wrapper.sh"
  }
  tool_path {
    name: "gcov"
    path: "/usr/bin/gcov"
  }
  tool_path {
    name: "ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "nm"
    path: "/usr/bin/nm"
  }
  tool_path {
    name: "objcopy"
    path: "/usr/bin/objcopy"
  }
  tool_path {
    name: "objdump"
    path: "/usr/bin/objdump"
  }
  tool_path {
    name: "strip"
    path: "/usr/bin/strip"
  }
  make_variable {
    name: "STACK_FRAME_UNLIMITED"
    value: "-Wframe-larger-than=100000000 -Wno-vla"
  }
  %{cxx_builtin_include_directory}
  builtin_sysroot: ""
  feature {
    name: "fastbuild"
  }
  feature {
    name: "no_legacy_features"
  }
  feature {
    name: "opt"
  }
  feature {
    name: "dbg"
  }
  feature {
    name: "link_libc++"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-lc++"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    enabled: true
  }
  feature {
    name: "compile_all_modules"
  }
  feature {
    name: "exclude_private_headers_in_module_maps"
  }
  feature {
    name: "has_configured_linker_path"
  }
  feature {
    name: "only_doth_headers_in_module_maps"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-fstack-protector"
        flag: "-fcolor-diagnostics"
        flag: "-Wall"
        flag: "-Wthread-safety"
        flag: "-Wself-assign"
        flag: "-fno-omit-frame-pointer"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-O0"
        flag: "-DDEBUG"
      }
      with_feature {
        feature: "fastbuild"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g0"
        flag: "-O2"
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-DNDEBUG"
        flag: "-ffunction-sections"
        flag: "-fdata-sections"
      }
      with_feature {
        feature: "opt"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g"
      }
      with_feature {
        feature: "dbg"
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
        flag: "-std=c++11"
      }
    }
    enabled: true
  }
  feature {
    name: "debug_prefix_map_pwd_is_dot"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "DEBUG_PREFIX_MAP_PWD=."
      }
    }
  }
  feature {
    name: "generate_dsym_file"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-g"
      }
    }
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "DSYM_HINT_LINKED_BINARY=%{linked_binary}"
        flag: "DSYM_HINT_DSYM_PATH=%{dsym_path}"
      }
    }
  }
  feature {
    name: "contains_objc_source"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fobjc-link-runtime"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "objc_actions"
    implies: "objc-compile"
    implies: "objc++-compile"
    implies: "objc-fully-link"
    implies: "objc-archive"
    implies: "objc-executable"
    implies: "objc++-executable"
    implies: "assemble"
    implies: "preprocess-assemble"
    implies: "c-compile"
    implies: "c++-compile"
    implies: "c++-link-static-library"
    implies: "c++-link-dynamic-library"
    implies: "c++-link-nodeps-dynamic-library"
    implies: "c++-link-executable"
  }
  feature {
    name: "strip_debug_symbols"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-Wl,-S"
        expand_if_all_available: "strip_debug_symbols"
      }
    }
  }
  feature {
    name: "symbol_counts"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,--print-symbol-counts=%{symbol_counts_output}"
        expand_if_all_available: "symbol_counts_output"
      }
    }
  }
  feature {
    name: "shared_flag"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-shared"
      }
    }
  }
  feature {
    name: "kernel_extension"
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
        flag: "-o"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "archiver_flags"
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "rcS"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "runtime_root_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-Wl,-rpath,@loader_path/%{runtime_library_search_directories}"
        iterate_over: "runtime_library_search_directories"
        expand_if_all_available: "runtime_library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_flags}"
        iterate_over: "runtime_root_flags"
        expand_if_all_available: "runtime_root_flags"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_entries}"
        iterate_over: "runtime_root_entries"
        expand_if_all_available: "runtime_root_entries"
      }
    }
  }
  feature {
    name: "input_param_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-L%{library_search_directories}"
        iterate_over: "library_search_directories"
        expand_if_all_available: "library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
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
        flag: "-Wl,-force_load,%{whole_archive_linker_params}"
        iterate_over: "whole_archive_linker_params"
        expand_if_all_available: "whole_archive_linker_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{linker_input_params}"
        iterate_over: "linker_input_params"
        expand_if_all_available: "linker_input_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag_group {
          flag: "-Wl,--start-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.object_files}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.object_files}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          iterate_over: "libraries_to_link.object_files"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag: "-Wl,--end-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
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
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "static_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "dynamic_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l:%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l:%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "versioned_dynamic_library"
          }
        }
        iterate_over: "libraries_to_link"
        expand_if_all_available: "libraries_to_link"
      }
    }
  }
  feature {
    name: "force_pic_flags"
    flag_set {
      action: "c++-link-executable"
      flag_group {
        flag: "-Wl,-pie"
        expand_if_all_available: "force_pic"
      }
    }
  }
  feature {
    name: "pch"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-include"
        flag: "%{pch_file}"
      }
    }
  }
  feature {
    name: "module_maps"
  }
  feature {
    name: "use_objc_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-name=%{module_name}"
        flag: "-iquote"
        flag: "%{module_maps_dir}"
        flag: "-fmodules-cache-path=%{modules_cache_path}"
      }
    }
  }
  feature {
    name: "no_enable_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-maps"
      }
    }
    requires {
      feature: "use_objc_modules"
    }
  }
  feature {
    name: "apply_default_warnings"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-Wshorten-64-to-32"
        flag: "-Wbool-conversion"
        flag: "-Wconstant-conversion"
        flag: "-Wduplicate-method-match"
        flag: "-Wempty-body"
        flag: "-Wenum-conversion"
        flag: "-Wint-conversion"
        flag: "-Wunreachable-code"
        flag: "-Wmismatched-return-types"
        flag: "-Wundeclared-selector"
        flag: "-Wuninitialized"
        flag: "-Wunused-function"
        flag: "-Wunused-variable"
      }
    }
  }
  feature {
    name: "includes"
    flag_set {
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "clif-match"
      flag_group {
        flag: "-include"
        flag: "%{includes}"
        iterate_over: "includes"
        expand_if_all_available: "includes"
      }
    }
    enabled: true
  }
  feature {
    name: "include_paths"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "clif-match"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-iquote"
        flag: "%{quote_include_paths}"
        iterate_over: "quote_include_paths"
      }
      flag_group {
        flag: "-I%{include_paths}"
        iterate_over: "include_paths"
      }
      flag_group {
        flag: "-isystem"
        flag: "%{system_include_paths}"
        iterate_over: "system_include_paths"
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
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "linkstamp-compile"
      action: "clif-match"
      flag_group {
        flag: "--sysroot=%{sysroot}"
        expand_if_all_available: "sysroot"
      }
    }
  }
  feature {
    name: "dependency_file"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      flag_group {
        flag: "-MD"
        flag: "-MF"
        flag: "%{dependency_file}"
        expand_if_all_available: "dependency_file"
      }
    }
  }
  feature {
    name: "pic"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "preprocess-assemble"
      flag_group {
        flag: "-fPIC"
        expand_if_all_available: "pic"
      }
    }
  }
  feature {
    name: "per_object_debug_info"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-gsplit-dwarf"
        expand_if_all_available: "per_object_debug_info_file"
      }
    }
  }
  feature {
    name: "preprocessor_defines"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D%{preprocessor_defines}"
        iterate_over: "preprocessor_defines"
      }
    }
  }
  feature {
    name: "framework_paths"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-F%{framework_paths}"
        iterate_over: "framework_paths"
      }
    }
  }
  feature {
    name: "random_seed"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      flag_group {
        flag: "-frandom-seed=%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "fdo_instrument"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fprofile-generate=%{fdo_instrument_path}"
        flag: "-fno-data-sections"
        expand_if_all_available: "fdo_instrument_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "fdo_optimize"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fprofile-use=%{fdo_profile_path}"
        flag: "-Xclang-only=-Wno-profile-instr-unprofiled"
        flag: "-Xclang-only=-Wno-profile-instr-out-of-date"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "autofdo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fauto-profile=%{fdo_profile_path}"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "lipo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fripa"
      }
    }
    requires {
      feature: "autofdo"
    }
    requires {
      feature: "fdo_optimize"
    }
    requires {
      feature: "fdo_instrument"
    }
  }
  feature {
    name: "coverage"
  }
  feature {
    name: "llvm_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-instr-generate"
        flag: "-fcoverage-mapping"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-fprofile-instr-generate"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "gcc_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-arcs"
        flag: "-ftest-coverage"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "--coverage"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "apply_default_compiler_flags"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-DOS_IOS"
        flag: "-fno-autolink"
      }
    }
  }
  feature {
    name: "include_system_dirs"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-isysroot"
        flag: "%{sdk_dir}"
      }
    }
  }
  feature {
    name: "bitcode_embedded"
  }
  feature {
    name: "bitcode_embedded_markers"
  }
  feature {
    name: "objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fobjc-arc"
        expand_if_all_available: "objc_arc"
      }
    }
  }
  feature {
    name: "no_objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fno-objc-arc"
        expand_if_all_available: "no_objc_arc"
      }
    }
  }
  feature {
    name: "apple_env"
    env_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-archive"
      action: "objc-fully-link"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      action: "objc-executable"
      action: "objc++-executable"
      action: "linkstamp-compile"
      env_entry {
        key: "XCODE_VERSION_OVERRIDE"
        value: "%{xcode_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_VERSION_OVERRIDE"
        value: "%{apple_sdk_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_PLATFORM"
        value: "%{apple_sdk_platform_value}"
      }
    }
  }
  feature {
    name: "user_link_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "%{user_link_flags}"
        iterate_over: "user_link_flags"
        expand_if_all_available: "user_link_flags"
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
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-headerpad_max_install_names"
        flag: "-no-canonical-prefixes"
        flag: "-target"
        flag: "i386-apple-ios"
      }
    }
    enabled: true
  }
  feature {
    name: "version_min"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-mios-simulator-version-min=%{version_min}"
      }
    }
  }
  feature {
    name: "dead_strip"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-dead_strip"
        flag: "-no_dead_strip_inits_and_terms"
      }
    }
    requires {
      feature: "opt"
    }
  }
  feature {
    name: "cpp_linker_flags"
  }
  feature {
    name: "apply_implicit_frameworks"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-framework"
        flag: "Foundation"
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "link_cocoa"
  }
  feature {
    name: "apply_simulator_compiler_flags"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fexceptions"
        flag: "-fasm-blocks"
        flag: "-fobjc-abi-version=2"
        flag: "-fobjc-legacy-dispatch"
      }
    }
  }
  feature {
    name: "unfiltered_cxx_flags"
  }
  feature {
    name: "user_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "%{user_compile_flags}"
        iterate_over: "user_compile_flags"
        expand_if_all_available: "user_compile_flags"
      }
    }
  }
  feature {
    name: "unfiltered_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      flag_group {
        flag: "-no-canonical-prefixes"
        flag: "-Wno-builtin-macro-redefined"
        flag: "-D__DATE__=\"redacted\""
        flag: "-D__TIMESTAMP__=\"redacted\""
        flag: "-D__TIME__=\"redacted\""
        flag: "-target"
        flag: "i386-apple-ios"
      }
    }
  }
  feature {
    name: "linker_param_file"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
      }
    }
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
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
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-c"
        flag: "%{source_file}"
        expand_if_all_available: "source_file"
      }
    }
  }
  feature {
    name: "compiler_output_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-S"
        expand_if_all_available: "output_assembly_file"
      }
      flag_group {
        flag: "-E"
        expand_if_all_available: "output_preprocess_file"
      }
      flag_group {
        flag: "-o"
        flag: "%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "objcopy_embed_flags"
    flag_set {
      action: "objcopy_embed_data"
      flag_group {
        flag: "-I"
        flag: "binary"
      }
    }
    enabled: true
  }
  action_config {
    config_name: "strip"
    action_name: "strip"
    tool {
      tool_path: "/usr/bin/strip"
    }
    flag_set {
      flag_group {
        flag: "-S"
        flag: "-o"
        flag: "%{output_file}"
      }
      flag_group {
        flag: "%{stripopts}"
        iterate_over: "stripopts"
      }
      flag_group {
        flag: "%{input_file}"
      }
    }
  }
  action_config {
    config_name: "c-compile"
    action_name: "c-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-compile"
    action_name: "c++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "linkstamp-compile"
    action_name: "linkstamp-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-module-compile"
    action_name: "c++-module-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-header-parsing"
    action_name: "c++-header-parsing"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-compile"
    action_name: "objc-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "i386"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "objc_actions"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "apply_simulator_compiler_flags"
  }
  action_config {
    config_name: "objc++-compile"
    action_name: "objc++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "i386"
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "apply_simulator_compiler_flags"
  }
  action_config {
    config_name: "assemble"
    action_name: "assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "include_system_dirs"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "preprocess-assemble"
    action_name: "preprocess-assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-archive"
    action_name: "objc-archive"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-filelist"
        flag: "%{obj_list_path}"
        flag: "-arch_only"
        flag: "i386"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{archive_path}"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-executable"
    action_name: "objc-executable"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "i386"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "objc++-executable"
    action_name: "objc++-executable"
    tool {
      tool_path: "wrapped_clang_pp"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
      flag_group {
        flag: "-arch"
        flag: "i386"
      }
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "c++-link-executable"
    action_name: "c++-link-executable"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "symbol_counts"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "force_pic_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-dynamic-library"
    action_name: "c++-link-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-nodeps-dynamic-library"
    action_name: "c++-link-nodeps-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-static-library"
    action_name: "c++-link-static-library"
    tool {
      tool_path: "wrapped_ar"
      execution_requirement: "requires-darwin"
    }
    implies: "runtime_root_flags"
    implies: "archiver_flags"
    implies: "input_param_flags"
    implies: "linker_param_file"
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-fully-link"
    action_name: "objc-fully-link"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-arch_only"
        flag: "i386"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{fully_linked_archive_path}"
      }
      flag_group {
        flag: "%{objc_library_exec_paths}"
        iterate_over: "objc_library_exec_paths"
      }
      flag_group {
        flag: "%{cc_library_exec_paths}"
        iterate_over: "cc_library_exec_paths"
      }
      flag_group {
        flag: "%{imported_library_exec_paths}"
        iterate_over: "imported_library_exec_paths"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objcopy_embed_data"
    action_name: "objcopy_embed_data"
    tool {
      tool_path: "/usr/bin/objcopy"
    }
    enabled: true
  }
  cc_target_os: "apple"
}
toolchain {
  toolchain_identifier: "ios_armv7"
  host_system_name: "x86_64-apple-macosx"
  target_system_name: "armv7-apple-ios"
  target_cpu: "ios_armv7"
  target_libc: "ios"
  compiler: "compiler"
  abi_version: "local"
  abi_libc_version: "local"
  tool_path {
    name: "ar"
    path: "wrapped_ar"
  }
  tool_path {
    name: "compat-ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "cpp"
    path: "/usr/bin/cpp"
  }
  tool_path {
    name: "dwp"
    path: "/usr/bin/dwp"
  }
  tool_path {
    name: "gcc"
    path: "cc_wrapper.sh"
  }
  tool_path {
    name: "gcov"
    path: "/usr/bin/gcov"
  }
  tool_path {
    name: "ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "nm"
    path: "/usr/bin/nm"
  }
  tool_path {
    name: "objcopy"
    path: "/usr/bin/objcopy"
  }
  tool_path {
    name: "objdump"
    path: "/usr/bin/objdump"
  }
  tool_path {
    name: "strip"
    path: "/usr/bin/strip"
  }
  make_variable {
    name: "STACK_FRAME_UNLIMITED"
    value: "-Wframe-larger-than=100000000 -Wno-vla"
  }
  %{cxx_builtin_include_directory}
  builtin_sysroot: ""
  feature {
    name: "fastbuild"
  }
  feature {
    name: "no_legacy_features"
  }
  feature {
    name: "opt"
  }
  feature {
    name: "dbg"
  }
  feature {
    name: "link_libc++"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-lc++"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    enabled: true
  }
  feature {
    name: "compile_all_modules"
  }
  feature {
    name: "exclude_private_headers_in_module_maps"
  }
  feature {
    name: "has_configured_linker_path"
  }
  feature {
    name: "only_doth_headers_in_module_maps"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-fstack-protector"
        flag: "-fcolor-diagnostics"
        flag: "-Wall"
        flag: "-Wthread-safety"
        flag: "-Wself-assign"
        flag: "-fno-omit-frame-pointer"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-O0"
        flag: "-DDEBUG"
      }
      with_feature {
        feature: "fastbuild"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g0"
        flag: "-O2"
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-DNDEBUG"
        flag: "-ffunction-sections"
        flag: "-fdata-sections"
      }
      with_feature {
        feature: "opt"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g"
      }
      with_feature {
        feature: "dbg"
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
        flag: "-std=c++11"
      }
    }
    enabled: true
  }
  feature {
    name: "debug_prefix_map_pwd_is_dot"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "DEBUG_PREFIX_MAP_PWD=."
      }
    }
  }
  feature {
    name: "generate_dsym_file"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-g"
      }
    }
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "DSYM_HINT_LINKED_BINARY=%{linked_binary}"
        flag: "DSYM_HINT_DSYM_PATH=%{dsym_path}"
      }
    }
  }
  feature {
    name: "contains_objc_source"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fobjc-link-runtime"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "objc_actions"
    implies: "objc-compile"
    implies: "objc++-compile"
    implies: "objc-fully-link"
    implies: "objc-archive"
    implies: "objc-executable"
    implies: "objc++-executable"
    implies: "assemble"
    implies: "preprocess-assemble"
    implies: "c-compile"
    implies: "c++-compile"
    implies: "c++-link-static-library"
    implies: "c++-link-dynamic-library"
    implies: "c++-link-nodeps-dynamic-library"
    implies: "c++-link-executable"
  }
  feature {
    name: "strip_debug_symbols"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-Wl,-S"
        expand_if_all_available: "strip_debug_symbols"
      }
    }
  }
  feature {
    name: "symbol_counts"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,--print-symbol-counts=%{symbol_counts_output}"
        expand_if_all_available: "symbol_counts_output"
      }
    }
  }
  feature {
    name: "shared_flag"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-shared"
      }
    }
  }
  feature {
    name: "kernel_extension"
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
        flag: "-o"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "archiver_flags"
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "rcS"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "runtime_root_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-Wl,-rpath,@loader_path/%{runtime_library_search_directories}"
        iterate_over: "runtime_library_search_directories"
        expand_if_all_available: "runtime_library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_flags}"
        iterate_over: "runtime_root_flags"
        expand_if_all_available: "runtime_root_flags"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_entries}"
        iterate_over: "runtime_root_entries"
        expand_if_all_available: "runtime_root_entries"
      }
    }
  }
  feature {
    name: "input_param_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-L%{library_search_directories}"
        iterate_over: "library_search_directories"
        expand_if_all_available: "library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
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
        flag: "-Wl,-force_load,%{whole_archive_linker_params}"
        iterate_over: "whole_archive_linker_params"
        expand_if_all_available: "whole_archive_linker_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{linker_input_params}"
        iterate_over: "linker_input_params"
        expand_if_all_available: "linker_input_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag_group {
          flag: "-Wl,--start-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.object_files}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.object_files}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          iterate_over: "libraries_to_link.object_files"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag: "-Wl,--end-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
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
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "static_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "dynamic_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l:%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l:%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "versioned_dynamic_library"
          }
        }
        iterate_over: "libraries_to_link"
        expand_if_all_available: "libraries_to_link"
      }
    }
  }
  feature {
    name: "force_pic_flags"
    flag_set {
      action: "c++-link-executable"
      flag_group {
        flag: "-Wl,-pie"
        expand_if_all_available: "force_pic"
      }
    }
  }
  feature {
    name: "pch"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-include"
        flag: "%{pch_file}"
      }
    }
  }
  feature {
    name: "module_maps"
  }
  feature {
    name: "use_objc_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-name=%{module_name}"
        flag: "-iquote"
        flag: "%{module_maps_dir}"
        flag: "-fmodules-cache-path=%{modules_cache_path}"
      }
    }
  }
  feature {
    name: "no_enable_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-maps"
      }
    }
    requires {
      feature: "use_objc_modules"
    }
  }
  feature {
    name: "apply_default_warnings"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-Wshorten-64-to-32"
        flag: "-Wbool-conversion"
        flag: "-Wconstant-conversion"
        flag: "-Wduplicate-method-match"
        flag: "-Wempty-body"
        flag: "-Wenum-conversion"
        flag: "-Wint-conversion"
        flag: "-Wunreachable-code"
        flag: "-Wmismatched-return-types"
        flag: "-Wundeclared-selector"
        flag: "-Wuninitialized"
        flag: "-Wunused-function"
        flag: "-Wunused-variable"
      }
    }
  }
  feature {
    name: "includes"
    flag_set {
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "clif-match"
      flag_group {
        flag: "-include"
        flag: "%{includes}"
        iterate_over: "includes"
        expand_if_all_available: "includes"
      }
    }
    enabled: true
  }
  feature {
    name: "include_paths"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "clif-match"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-iquote"
        flag: "%{quote_include_paths}"
        iterate_over: "quote_include_paths"
      }
      flag_group {
        flag: "-I%{include_paths}"
        iterate_over: "include_paths"
      }
      flag_group {
        flag: "-isystem"
        flag: "%{system_include_paths}"
        iterate_over: "system_include_paths"
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
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "linkstamp-compile"
      action: "clif-match"
      flag_group {
        flag: "--sysroot=%{sysroot}"
        expand_if_all_available: "sysroot"
      }
    }
  }
  feature {
    name: "dependency_file"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      flag_group {
        flag: "-MD"
        flag: "-MF"
        flag: "%{dependency_file}"
        expand_if_all_available: "dependency_file"
      }
    }
  }
  feature {
    name: "pic"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "preprocess-assemble"
      flag_group {
        flag: "-fPIC"
        expand_if_all_available: "pic"
      }
    }
  }
  feature {
    name: "per_object_debug_info"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-gsplit-dwarf"
        expand_if_all_available: "per_object_debug_info_file"
      }
    }
  }
  feature {
    name: "preprocessor_defines"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D%{preprocessor_defines}"
        iterate_over: "preprocessor_defines"
      }
    }
  }
  feature {
    name: "framework_paths"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-F%{framework_paths}"
        iterate_over: "framework_paths"
      }
    }
  }
  feature {
    name: "random_seed"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      flag_group {
        flag: "-frandom-seed=%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "fdo_instrument"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fprofile-generate=%{fdo_instrument_path}"
        flag: "-fno-data-sections"
        expand_if_all_available: "fdo_instrument_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "fdo_optimize"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fprofile-use=%{fdo_profile_path}"
        flag: "-Xclang-only=-Wno-profile-instr-unprofiled"
        flag: "-Xclang-only=-Wno-profile-instr-out-of-date"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "autofdo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fauto-profile=%{fdo_profile_path}"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "lipo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fripa"
      }
    }
    requires {
      feature: "autofdo"
    }
    requires {
      feature: "fdo_optimize"
    }
    requires {
      feature: "fdo_instrument"
    }
  }
  feature {
    name: "coverage"
  }
  feature {
    name: "llvm_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-instr-generate"
        flag: "-fcoverage-mapping"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-fprofile-instr-generate"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "gcc_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-arcs"
        flag: "-ftest-coverage"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "--coverage"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "apply_default_compiler_flags"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-DOS_IOS"
        flag: "-fno-autolink"
      }
    }
  }
  feature {
    name: "include_system_dirs"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-isysroot"
        flag: "%{sdk_dir}"
      }
    }
  }
  feature {
    name: "bitcode_embedded"
  }
  feature {
    name: "bitcode_embedded_markers"
  }
  feature {
    name: "objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fobjc-arc"
        expand_if_all_available: "objc_arc"
      }
    }
  }
  feature {
    name: "no_objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fno-objc-arc"
        expand_if_all_available: "no_objc_arc"
      }
    }
  }
  feature {
    name: "apple_env"
    env_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-archive"
      action: "objc-fully-link"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      action: "objc-executable"
      action: "objc++-executable"
      action: "linkstamp-compile"
      env_entry {
        key: "XCODE_VERSION_OVERRIDE"
        value: "%{xcode_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_VERSION_OVERRIDE"
        value: "%{apple_sdk_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_PLATFORM"
        value: "%{apple_sdk_platform_value}"
      }
    }
  }
  feature {
    name: "user_link_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "%{user_link_flags}"
        iterate_over: "user_link_flags"
        expand_if_all_available: "user_link_flags"
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
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-headerpad_max_install_names"
        flag: "-no-canonical-prefixes"
        flag: "-target"
        flag: "armv7-apple-ios"
      }
    }
    enabled: true
  }
  feature {
    name: "version_min"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-miphoneos-version-min=%{version_min}"
      }
    }
  }
  feature {
    name: "dead_strip"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-dead_strip"
        flag: "-no_dead_strip_inits_and_terms"
      }
    }
    requires {
      feature: "opt"
    }
  }
  feature {
    name: "cpp_linker_flags"
  }
  feature {
    name: "apply_implicit_frameworks"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-framework"
        flag: "Foundation"
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "link_cocoa"
  }
  feature {
    name: "apply_simulator_compiler_flags"
  }
  feature {
    name: "unfiltered_cxx_flags"
  }
  feature {
    name: "user_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "%{user_compile_flags}"
        iterate_over: "user_compile_flags"
        expand_if_all_available: "user_compile_flags"
      }
    }
  }
  feature {
    name: "unfiltered_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      flag_group {
        flag: "-no-canonical-prefixes"
        flag: "-Wno-builtin-macro-redefined"
        flag: "-D__DATE__=\"redacted\""
        flag: "-D__TIMESTAMP__=\"redacted\""
        flag: "-D__TIME__=\"redacted\""
        flag: "-target"
        flag: "armv7-apple-ios"
      }
    }
  }
  feature {
    name: "linker_param_file"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
      }
    }
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
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
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-c"
        flag: "%{source_file}"
        expand_if_all_available: "source_file"
      }
    }
  }
  feature {
    name: "compiler_output_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-S"
        expand_if_all_available: "output_assembly_file"
      }
      flag_group {
        flag: "-E"
        expand_if_all_available: "output_preprocess_file"
      }
      flag_group {
        flag: "-o"
        flag: "%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "objcopy_embed_flags"
    flag_set {
      action: "objcopy_embed_data"
      flag_group {
        flag: "-I"
        flag: "binary"
      }
    }
    enabled: true
  }
  action_config {
    config_name: "strip"
    action_name: "strip"
    tool {
      tool_path: "/usr/bin/strip"
    }
    flag_set {
      flag_group {
        flag: "-S"
        flag: "-o"
        flag: "%{output_file}"
      }
      flag_group {
        flag: "%{stripopts}"
        iterate_over: "stripopts"
      }
      flag_group {
        flag: "%{input_file}"
      }
    }
  }
  action_config {
    config_name: "c-compile"
    action_name: "c-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-compile"
    action_name: "c++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "linkstamp-compile"
    action_name: "linkstamp-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-module-compile"
    action_name: "c++-module-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-header-parsing"
    action_name: "c++-header-parsing"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-compile"
    action_name: "objc-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "armv7"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "objc_actions"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
  }
  action_config {
    config_name: "objc++-compile"
    action_name: "objc++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "armv7"
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
  }
  action_config {
    config_name: "assemble"
    action_name: "assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "include_system_dirs"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "preprocess-assemble"
    action_name: "preprocess-assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-archive"
    action_name: "objc-archive"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-filelist"
        flag: "%{obj_list_path}"
        flag: "-arch_only"
        flag: "armv7"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{archive_path}"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-executable"
    action_name: "objc-executable"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "armv7"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "objc++-executable"
    action_name: "objc++-executable"
    tool {
      tool_path: "wrapped_clang_pp"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
      flag_group {
        flag: "-arch"
        flag: "armv7"
      }
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "c++-link-executable"
    action_name: "c++-link-executable"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "symbol_counts"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "force_pic_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-dynamic-library"
    action_name: "c++-link-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-nodeps-dynamic-library"
    action_name: "c++-link-nodeps-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-static-library"
    action_name: "c++-link-static-library"
    tool {
      tool_path: "wrapped_ar"
      execution_requirement: "requires-darwin"
    }
    implies: "runtime_root_flags"
    implies: "archiver_flags"
    implies: "input_param_flags"
    implies: "linker_param_file"
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-fully-link"
    action_name: "objc-fully-link"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-arch_only"
        flag: "armv7"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{fully_linked_archive_path}"
      }
      flag_group {
        flag: "%{objc_library_exec_paths}"
        iterate_over: "objc_library_exec_paths"
      }
      flag_group {
        flag: "%{cc_library_exec_paths}"
        iterate_over: "cc_library_exec_paths"
      }
      flag_group {
        flag: "%{imported_library_exec_paths}"
        iterate_over: "imported_library_exec_paths"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objcopy_embed_data"
    action_name: "objcopy_embed_data"
    tool {
      tool_path: "/usr/bin/objcopy"
    }
    enabled: true
  }
  cc_target_os: "apple"
}
toolchain {
  toolchain_identifier: "watchos_armv7k"
  host_system_name: "x86_64-apple-macosx"
  target_system_name: "armv7-apple-watchos"
  target_cpu: "watchos_armv7k"
  target_libc: "watchos"
  compiler: "compiler"
  abi_version: "local"
  abi_libc_version: "local"
  tool_path {
    name: "ar"
    path: "wrapped_ar"
  }
  tool_path {
    name: "compat-ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "cpp"
    path: "/usr/bin/cpp"
  }
  tool_path {
    name: "dwp"
    path: "/usr/bin/dwp"
  }
  tool_path {
    name: "gcc"
    path: "cc_wrapper.sh"
  }
  tool_path {
    name: "gcov"
    path: "/usr/bin/gcov"
  }
  tool_path {
    name: "ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "nm"
    path: "/usr/bin/nm"
  }
  tool_path {
    name: "objcopy"
    path: "/usr/bin/objcopy"
  }
  tool_path {
    name: "objdump"
    path: "/usr/bin/objdump"
  }
  tool_path {
    name: "strip"
    path: "/usr/bin/strip"
  }
  make_variable {
    name: "STACK_FRAME_UNLIMITED"
    value: "-Wframe-larger-than=100000000 -Wno-vla"
  }
  %{cxx_builtin_include_directory}
  builtin_sysroot: ""
  feature {
    name: "fastbuild"
  }
  feature {
    name: "no_legacy_features"
  }
  feature {
    name: "opt"
  }
  feature {
    name: "dbg"
  }
  feature {
    name: "link_libc++"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-lc++"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    enabled: true
  }
  feature {
    name: "compile_all_modules"
  }
  feature {
    name: "exclude_private_headers_in_module_maps"
  }
  feature {
    name: "has_configured_linker_path"
  }
  feature {
    name: "only_doth_headers_in_module_maps"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-fstack-protector"
        flag: "-fcolor-diagnostics"
        flag: "-Wall"
        flag: "-Wthread-safety"
        flag: "-Wself-assign"
        flag: "-fno-omit-frame-pointer"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-O0"
        flag: "-DDEBUG"
      }
      with_feature {
        feature: "fastbuild"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g0"
        flag: "-O2"
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-DNDEBUG"
        flag: "-ffunction-sections"
        flag: "-fdata-sections"
      }
      with_feature {
        feature: "opt"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g"
      }
      with_feature {
        feature: "dbg"
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
        flag: "-std=c++11"
      }
    }
    enabled: true
  }
  feature {
    name: "debug_prefix_map_pwd_is_dot"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "DEBUG_PREFIX_MAP_PWD=."
      }
    }
  }
  feature {
    name: "generate_dsym_file"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-g"
      }
    }
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "DSYM_HINT_LINKED_BINARY=%{linked_binary}"
        flag: "DSYM_HINT_DSYM_PATH=%{dsym_path}"
      }
    }
  }
  feature {
    name: "contains_objc_source"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fobjc-link-runtime"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "objc_actions"
    implies: "objc-compile"
    implies: "objc++-compile"
    implies: "objc-fully-link"
    implies: "objc-archive"
    implies: "objc-executable"
    implies: "objc++-executable"
    implies: "assemble"
    implies: "preprocess-assemble"
    implies: "c-compile"
    implies: "c++-compile"
    implies: "c++-link-static-library"
    implies: "c++-link-dynamic-library"
    implies: "c++-link-nodeps-dynamic-library"
    implies: "c++-link-executable"
  }
  feature {
    name: "strip_debug_symbols"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-Wl,-S"
        expand_if_all_available: "strip_debug_symbols"
      }
    }
  }
  feature {
    name: "symbol_counts"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,--print-symbol-counts=%{symbol_counts_output}"
        expand_if_all_available: "symbol_counts_output"
      }
    }
  }
  feature {
    name: "shared_flag"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-shared"
      }
    }
  }
  feature {
    name: "kernel_extension"
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
        flag: "-o"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "archiver_flags"
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "rcS"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "runtime_root_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-Wl,-rpath,@loader_path/%{runtime_library_search_directories}"
        iterate_over: "runtime_library_search_directories"
        expand_if_all_available: "runtime_library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_flags}"
        iterate_over: "runtime_root_flags"
        expand_if_all_available: "runtime_root_flags"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_entries}"
        iterate_over: "runtime_root_entries"
        expand_if_all_available: "runtime_root_entries"
      }
    }
  }
  feature {
    name: "input_param_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-L%{library_search_directories}"
        iterate_over: "library_search_directories"
        expand_if_all_available: "library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
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
        flag: "-Wl,-force_load,%{whole_archive_linker_params}"
        iterate_over: "whole_archive_linker_params"
        expand_if_all_available: "whole_archive_linker_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{linker_input_params}"
        iterate_over: "linker_input_params"
        expand_if_all_available: "linker_input_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag_group {
          flag: "-Wl,--start-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.object_files}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.object_files}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          iterate_over: "libraries_to_link.object_files"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag: "-Wl,--end-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
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
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "static_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "dynamic_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l:%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l:%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "versioned_dynamic_library"
          }
        }
        iterate_over: "libraries_to_link"
        expand_if_all_available: "libraries_to_link"
      }
    }
  }
  feature {
    name: "force_pic_flags"
    flag_set {
      action: "c++-link-executable"
      flag_group {
        flag: "-Wl,-pie"
        expand_if_all_available: "force_pic"
      }
    }
  }
  feature {
    name: "pch"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-include"
        flag: "%{pch_file}"
      }
    }
  }
  feature {
    name: "module_maps"
  }
  feature {
    name: "use_objc_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-name=%{module_name}"
        flag: "-iquote"
        flag: "%{module_maps_dir}"
        flag: "-fmodules-cache-path=%{modules_cache_path}"
      }
    }
  }
  feature {
    name: "no_enable_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-maps"
      }
    }
    requires {
      feature: "use_objc_modules"
    }
  }
  feature {
    name: "apply_default_warnings"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-Wshorten-64-to-32"
        flag: "-Wbool-conversion"
        flag: "-Wconstant-conversion"
        flag: "-Wduplicate-method-match"
        flag: "-Wempty-body"
        flag: "-Wenum-conversion"
        flag: "-Wint-conversion"
        flag: "-Wunreachable-code"
        flag: "-Wmismatched-return-types"
        flag: "-Wundeclared-selector"
        flag: "-Wuninitialized"
        flag: "-Wunused-function"
        flag: "-Wunused-variable"
      }
    }
  }
  feature {
    name: "includes"
    flag_set {
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "clif-match"
      flag_group {
        flag: "-include"
        flag: "%{includes}"
        iterate_over: "includes"
        expand_if_all_available: "includes"
      }
    }
    enabled: true
  }
  feature {
    name: "include_paths"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "clif-match"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-iquote"
        flag: "%{quote_include_paths}"
        iterate_over: "quote_include_paths"
      }
      flag_group {
        flag: "-I%{include_paths}"
        iterate_over: "include_paths"
      }
      flag_group {
        flag: "-isystem"
        flag: "%{system_include_paths}"
        iterate_over: "system_include_paths"
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
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "linkstamp-compile"
      action: "clif-match"
      flag_group {
        flag: "--sysroot=%{sysroot}"
        expand_if_all_available: "sysroot"
      }
    }
  }
  feature {
    name: "dependency_file"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      flag_group {
        flag: "-MD"
        flag: "-MF"
        flag: "%{dependency_file}"
        expand_if_all_available: "dependency_file"
      }
    }
  }
  feature {
    name: "pic"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "preprocess-assemble"
      flag_group {
        flag: "-fPIC"
        expand_if_all_available: "pic"
      }
    }
  }
  feature {
    name: "per_object_debug_info"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-gsplit-dwarf"
        expand_if_all_available: "per_object_debug_info_file"
      }
    }
  }
  feature {
    name: "preprocessor_defines"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D%{preprocessor_defines}"
        iterate_over: "preprocessor_defines"
      }
    }
  }
  feature {
    name: "framework_paths"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-F%{framework_paths}"
        iterate_over: "framework_paths"
      }
    }
  }
  feature {
    name: "random_seed"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      flag_group {
        flag: "-frandom-seed=%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "fdo_instrument"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fprofile-generate=%{fdo_instrument_path}"
        flag: "-fno-data-sections"
        expand_if_all_available: "fdo_instrument_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "fdo_optimize"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fprofile-use=%{fdo_profile_path}"
        flag: "-Xclang-only=-Wno-profile-instr-unprofiled"
        flag: "-Xclang-only=-Wno-profile-instr-out-of-date"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "autofdo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fauto-profile=%{fdo_profile_path}"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "lipo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fripa"
      }
    }
    requires {
      feature: "autofdo"
    }
    requires {
      feature: "fdo_optimize"
    }
    requires {
      feature: "fdo_instrument"
    }
  }
  feature {
    name: "coverage"
  }
  feature {
    name: "llvm_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-instr-generate"
        flag: "-fcoverage-mapping"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-fprofile-instr-generate"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "gcc_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-arcs"
        flag: "-ftest-coverage"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "--coverage"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "apply_default_compiler_flags"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-DOS_IOS"
        flag: "-fno-autolink"
      }
    }
  }
  feature {
    name: "include_system_dirs"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-isysroot"
        flag: "%{sdk_dir}"
        flag: "-F%{sdk_framework_dir}"
        flag: "-F%{platform_developer_framework_dir}"
      }
    }
  }
  feature {
    name: "bitcode_embedded"
  }
  feature {
    name: "bitcode_embedded_markers"
  }
  feature {
    name: "objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fobjc-arc"
        expand_if_all_available: "objc_arc"
      }
    }
  }
  feature {
    name: "no_objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fno-objc-arc"
        expand_if_all_available: "no_objc_arc"
      }
    }
  }
  feature {
    name: "apple_env"
    env_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-archive"
      action: "objc-fully-link"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      action: "objc-executable"
      action: "objc++-executable"
      action: "linkstamp-compile"
      env_entry {
        key: "XCODE_VERSION_OVERRIDE"
        value: "%{xcode_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_VERSION_OVERRIDE"
        value: "%{apple_sdk_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_PLATFORM"
        value: "%{apple_sdk_platform_value}"
      }
    }
  }
  feature {
    name: "user_link_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "%{user_link_flags}"
        iterate_over: "user_link_flags"
        expand_if_all_available: "user_link_flags"
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
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-headerpad_max_install_names"
        flag: "-no-canonical-prefixes"
        flag: "-target"
        flag: "armv7-apple-watchos"
      }
    }
    enabled: true
  }
  feature {
    name: "version_min"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-mwatchos-version-min=%{version_min}"
      }
    }
  }
  feature {
    name: "dead_strip"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-dead_strip"
        flag: "-no_dead_strip_inits_and_terms"
      }
    }
    requires {
      feature: "opt"
    }
  }
  feature {
    name: "cpp_linker_flags"
  }
  feature {
    name: "apply_implicit_frameworks"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-framework"
        flag: "Foundation"
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "link_cocoa"
  }
  feature {
    name: "apply_simulator_compiler_flags"
  }
  feature {
    name: "unfiltered_cxx_flags"
  }
  feature {
    name: "user_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "%{user_compile_flags}"
        iterate_over: "user_compile_flags"
        expand_if_all_available: "user_compile_flags"
      }
    }
  }
  feature {
    name: "unfiltered_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      flag_group {
        flag: "-no-canonical-prefixes"
        flag: "-Wno-builtin-macro-redefined"
        flag: "-D__DATE__=\"redacted\""
        flag: "-D__TIMESTAMP__=\"redacted\""
        flag: "-D__TIME__=\"redacted\""
        flag: "-target"
        flag: "armv7k-apple-watchos"
      }
    }
  }
  feature {
    name: "linker_param_file"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
      }
    }
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
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
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-c"
        flag: "%{source_file}"
        expand_if_all_available: "source_file"
      }
    }
  }
  feature {
    name: "compiler_output_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-S"
        expand_if_all_available: "output_assembly_file"
      }
      flag_group {
        flag: "-E"
        expand_if_all_available: "output_preprocess_file"
      }
      flag_group {
        flag: "-o"
        flag: "%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "objcopy_embed_flags"
    flag_set {
      action: "objcopy_embed_data"
      flag_group {
        flag: "-I"
        flag: "binary"
      }
    }
    enabled: true
  }
  action_config {
    config_name: "strip"
    action_name: "strip"
    tool {
      tool_path: "/usr/bin/strip"
    }
    flag_set {
      flag_group {
        flag: "-S"
        flag: "-o"
        flag: "%{output_file}"
      }
      flag_group {
        flag: "%{stripopts}"
        iterate_over: "stripopts"
      }
      flag_group {
        flag: "%{input_file}"
      }
    }
  }
  action_config {
    config_name: "c-compile"
    action_name: "c-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-compile"
    action_name: "c++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "linkstamp-compile"
    action_name: "linkstamp-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-module-compile"
    action_name: "c++-module-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-header-parsing"
    action_name: "c++-header-parsing"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-compile"
    action_name: "objc-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "armv7k"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "objc_actions"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
  }
  action_config {
    config_name: "objc++-compile"
    action_name: "objc++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "armv7k"
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
  }
  action_config {
    config_name: "assemble"
    action_name: "assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "include_system_dirs"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "preprocess-assemble"
    action_name: "preprocess-assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-archive"
    action_name: "objc-archive"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-filelist"
        flag: "%{obj_list_path}"
        flag: "-arch_only"
        flag: "armv7k"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{archive_path}"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-executable"
    action_name: "objc-executable"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "armv7k"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "objc++-executable"
    action_name: "objc++-executable"
    tool {
      tool_path: "wrapped_clang_pp"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
      flag_group {
        flag: "-arch"
        flag: "armv7k"
      }
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "c++-link-executable"
    action_name: "c++-link-executable"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "symbol_counts"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "force_pic_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-dynamic-library"
    action_name: "c++-link-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-nodeps-dynamic-library"
    action_name: "c++-link-nodeps-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-static-library"
    action_name: "c++-link-static-library"
    tool {
      tool_path: "wrapped_ar"
      execution_requirement: "requires-darwin"
    }
    implies: "runtime_root_flags"
    implies: "archiver_flags"
    implies: "input_param_flags"
    implies: "linker_param_file"
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-fully-link"
    action_name: "objc-fully-link"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-arch_only"
        flag: "armv7k"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{fully_linked_archive_path}"
      }
      flag_group {
        flag: "%{objc_library_exec_paths}"
        iterate_over: "objc_library_exec_paths"
      }
      flag_group {
        flag: "%{cc_library_exec_paths}"
        iterate_over: "cc_library_exec_paths"
      }
      flag_group {
        flag: "%{imported_library_exec_paths}"
        iterate_over: "imported_library_exec_paths"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objcopy_embed_data"
    action_name: "objcopy_embed_data"
    tool {
      tool_path: "/usr/bin/objcopy"
    }
    enabled: true
  }
  cc_target_os: "apple"
}
toolchain {
  toolchain_identifier: "tvos_arm64"
  host_system_name: "x86_64-apple-macosx"
  target_system_name: "arm64-apple-tvos"
  target_cpu: "tvos_arm64"
  target_libc: "tvos"
  compiler: "compiler"
  abi_version: "local"
  abi_libc_version: "local"
  tool_path {
    name: "ar"
    path: "wrapped_ar"
  }
  tool_path {
    name: "compat-ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "cpp"
    path: "/usr/bin/cpp"
  }
  tool_path {
    name: "dwp"
    path: "/usr/bin/dwp"
  }
  tool_path {
    name: "gcc"
    path: "cc_wrapper.sh"
  }
  tool_path {
    name: "gcov"
    path: "/usr/bin/gcov"
  }
  tool_path {
    name: "ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "nm"
    path: "/usr/bin/nm"
  }
  tool_path {
    name: "objcopy"
    path: "/usr/bin/objcopy"
  }
  tool_path {
    name: "objdump"
    path: "/usr/bin/objdump"
  }
  tool_path {
    name: "strip"
    path: "/usr/bin/strip"
  }
  make_variable {
    name: "STACK_FRAME_UNLIMITED"
    value: "-Wframe-larger-than=100000000 -Wno-vla"
  }
  %{cxx_builtin_include_directory}
  builtin_sysroot: ""
  feature {
    name: "fastbuild"
  }
  feature {
    name: "no_legacy_features"
  }
  feature {
    name: "opt"
  }
  feature {
    name: "dbg"
  }
  feature {
    name: "link_libc++"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-lc++"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    enabled: true
  }
  feature {
    name: "compile_all_modules"
  }
  feature {
    name: "exclude_private_headers_in_module_maps"
  }
  feature {
    name: "has_configured_linker_path"
  }
  feature {
    name: "only_doth_headers_in_module_maps"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-fstack-protector"
        flag: "-fcolor-diagnostics"
        flag: "-Wall"
        flag: "-Wthread-safety"
        flag: "-Wself-assign"
        flag: "-fno-omit-frame-pointer"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-O0"
        flag: "-DDEBUG"
      }
      with_feature {
        feature: "fastbuild"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g0"
        flag: "-O2"
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-DNDEBUG"
        flag: "-ffunction-sections"
        flag: "-fdata-sections"
        flag: "-DNS_BLOCK_ASSERTIONS=1"
      }
      with_feature {
        feature: "opt"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g"
      }
      with_feature {
        feature: "dbg"
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
        flag: "-std=c++11"
      }
    }
    enabled: true
  }
  feature {
    name: "debug_prefix_map_pwd_is_dot"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "DEBUG_PREFIX_MAP_PWD=."
      }
    }
  }
  feature {
    name: "generate_dsym_file"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-g"
      }
    }
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "DSYM_HINT_LINKED_BINARY=%{linked_binary}"
        flag: "DSYM_HINT_DSYM_PATH=%{dsym_path}"
      }
    }
  }
  feature {
    name: "contains_objc_source"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fobjc-link-runtime"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "objc_actions"
    implies: "objc-compile"
    implies: "objc++-compile"
    implies: "objc-fully-link"
    implies: "objc-archive"
    implies: "objc-executable"
    implies: "objc++-executable"
    implies: "assemble"
    implies: "preprocess-assemble"
    implies: "c-compile"
    implies: "c++-compile"
    implies: "c++-link-static-library"
    implies: "c++-link-dynamic-library"
    implies: "c++-link-nodeps-dynamic-library"
    implies: "c++-link-executable"
  }
  feature {
    name: "strip_debug_symbols"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-Wl,-S"
        expand_if_all_available: "strip_debug_symbols"
      }
    }
  }
  feature {
    name: "symbol_counts"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,--print-symbol-counts=%{symbol_counts_output}"
        expand_if_all_available: "symbol_counts_output"
      }
    }
  }
  feature {
    name: "shared_flag"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-shared"
      }
    }
  }
  feature {
    name: "kernel_extension"
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
        flag: "-o"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "archiver_flags"
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "rcS"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "runtime_root_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-Wl,-rpath,@loader_path/%{runtime_library_search_directories}"
        iterate_over: "runtime_library_search_directories"
        expand_if_all_available: "runtime_library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_flags}"
        iterate_over: "runtime_root_flags"
        expand_if_all_available: "runtime_root_flags"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_entries}"
        iterate_over: "runtime_root_entries"
        expand_if_all_available: "runtime_root_entries"
      }
    }
  }
  feature {
    name: "input_param_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-L%{library_search_directories}"
        iterate_over: "library_search_directories"
        expand_if_all_available: "library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
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
        flag: "-Wl,-force_load,%{whole_archive_linker_params}"
        iterate_over: "whole_archive_linker_params"
        expand_if_all_available: "whole_archive_linker_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{linker_input_params}"
        iterate_over: "linker_input_params"
        expand_if_all_available: "linker_input_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag_group {
          flag: "-Wl,--start-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.object_files}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.object_files}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          iterate_over: "libraries_to_link.object_files"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag: "-Wl,--end-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
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
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "static_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "dynamic_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l:%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l:%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "versioned_dynamic_library"
          }
        }
        iterate_over: "libraries_to_link"
        expand_if_all_available: "libraries_to_link"
      }
    }
  }
  feature {
    name: "force_pic_flags"
    flag_set {
      action: "c++-link-executable"
      flag_group {
        flag: "-Wl,-pie"
        expand_if_all_available: "force_pic"
      }
    }
  }
  feature {
    name: "pch"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-include"
        flag: "%{pch_file}"
      }
    }
  }
  feature {
    name: "module_maps"
  }
  feature {
    name: "use_objc_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-name=%{module_name}"
        flag: "-iquote"
        flag: "%{module_maps_dir}"
        flag: "-fmodules-cache-path=%{modules_cache_path}"
      }
    }
  }
  feature {
    name: "no_enable_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-maps"
      }
    }
    requires {
      feature: "use_objc_modules"
    }
  }
  feature {
    name: "apply_default_warnings"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-Wshorten-64-to-32"
        flag: "-Wbool-conversion"
        flag: "-Wconstant-conversion"
        flag: "-Wduplicate-method-match"
        flag: "-Wempty-body"
        flag: "-Wenum-conversion"
        flag: "-Wint-conversion"
        flag: "-Wunreachable-code"
        flag: "-Wmismatched-return-types"
        flag: "-Wundeclared-selector"
        flag: "-Wuninitialized"
        flag: "-Wunused-function"
        flag: "-Wunused-variable"
      }
    }
  }
  feature {
    name: "includes"
    flag_set {
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "clif-match"
      flag_group {
        flag: "-include"
        flag: "%{includes}"
        iterate_over: "includes"
        expand_if_all_available: "includes"
      }
    }
    enabled: true
  }
  feature {
    name: "include_paths"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "clif-match"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-iquote"
        flag: "%{quote_include_paths}"
        iterate_over: "quote_include_paths"
      }
      flag_group {
        flag: "-I%{include_paths}"
        iterate_over: "include_paths"
      }
      flag_group {
        flag: "-isystem"
        flag: "%{system_include_paths}"
        iterate_over: "system_include_paths"
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
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "linkstamp-compile"
      action: "clif-match"
      flag_group {
        flag: "--sysroot=%{sysroot}"
        expand_if_all_available: "sysroot"
      }
    }
  }
  feature {
    name: "dependency_file"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      flag_group {
        flag: "-MD"
        flag: "-MF"
        flag: "%{dependency_file}"
        expand_if_all_available: "dependency_file"
      }
    }
  }
  feature {
    name: "pic"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "preprocess-assemble"
      flag_group {
        flag: "-fPIC"
        expand_if_all_available: "pic"
      }
    }
  }
  feature {
    name: "per_object_debug_info"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-gsplit-dwarf"
        expand_if_all_available: "per_object_debug_info_file"
      }
    }
  }
  feature {
    name: "preprocessor_defines"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D%{preprocessor_defines}"
        iterate_over: "preprocessor_defines"
      }
    }
  }
  feature {
    name: "framework_paths"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-F%{framework_paths}"
        iterate_over: "framework_paths"
      }
    }
  }
  feature {
    name: "random_seed"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      flag_group {
        flag: "-frandom-seed=%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "fdo_instrument"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fprofile-generate=%{fdo_instrument_path}"
        flag: "-fno-data-sections"
        expand_if_all_available: "fdo_instrument_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "fdo_optimize"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fprofile-use=%{fdo_profile_path}"
        flag: "-Xclang-only=-Wno-profile-instr-unprofiled"
        flag: "-Xclang-only=-Wno-profile-instr-out-of-date"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "autofdo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fauto-profile=%{fdo_profile_path}"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "lipo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fripa"
      }
    }
    requires {
      feature: "autofdo"
    }
    requires {
      feature: "fdo_optimize"
    }
    requires {
      feature: "fdo_instrument"
    }
  }
  feature {
    name: "coverage"
  }
  feature {
    name: "llvm_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-instr-generate"
        flag: "-fcoverage-mapping"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-fprofile-instr-generate"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "gcc_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-arcs"
        flag: "-ftest-coverage"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "--coverage"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "apply_default_compiler_flags"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-DOS_TVOS"
        flag: "-fno-autolink"
      }
    }
  }
  feature {
    name: "include_system_dirs"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-isysroot"
        flag: "%{sdk_dir}"
      }
    }
  }
  feature {
    name: "bitcode_embedded"
  }
  feature {
    name: "bitcode_embedded_markers"
  }
  feature {
    name: "objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fobjc-arc"
        expand_if_all_available: "objc_arc"
      }
    }
  }
  feature {
    name: "no_objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fno-objc-arc"
        expand_if_all_available: "no_objc_arc"
      }
    }
  }
  feature {
    name: "apple_env"
    env_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-archive"
      action: "objc-fully-link"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      action: "objc-executable"
      action: "objc++-executable"
      action: "linkstamp-compile"
      env_entry {
        key: "XCODE_VERSION_OVERRIDE"
        value: "%{xcode_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_VERSION_OVERRIDE"
        value: "%{apple_sdk_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_PLATFORM"
        value: "%{apple_sdk_platform_value}"
      }
    }
  }
  feature {
    name: "user_link_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "%{user_link_flags}"
        iterate_over: "user_link_flags"
        expand_if_all_available: "user_link_flags"
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
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-headerpad_max_install_names"
        flag: "-no-canonical-prefixes"
        flag: "-target"
        flag: "arm64-apple-tvos"
      }
    }
    enabled: true
  }
  feature {
    name: "version_min"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-mtvos-version-min=%{version_min}"
      }
    }
  }
  feature {
    name: "dead_strip"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-dead_strip"
        flag: "-no_dead_strip_inits_and_terms"
      }
    }
    requires {
      feature: "opt"
    }
  }
  feature {
    name: "cpp_linker_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      flag_group {
        flag: "-lc++"
        flag: "-target"
        flag: "arm64-apple-tvos"
      }
    }
  }
  feature {
    name: "apply_implicit_frameworks"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-framework"
        flag: "Foundation"
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "link_cocoa"
  }
  feature {
    name: "apply_simulator_compiler_flags"
  }
  feature {
    name: "unfiltered_cxx_flags"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-no-canonical-prefixes"
        flag: "-pthread"
      }
    }
  }
  feature {
    name: "user_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "%{user_compile_flags}"
        iterate_over: "user_compile_flags"
        expand_if_all_available: "user_compile_flags"
      }
    }
  }
  feature {
    name: "unfiltered_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      flag_group {
        flag: "-no-canonical-prefixes"
        flag: "-Wno-builtin-macro-redefined"
        flag: "-D__DATE__=\"redacted\""
        flag: "-D__TIMESTAMP__=\"redacted\""
        flag: "-D__TIME__=\"redacted\""
        flag: "-target"
        flag: "arm64-apple-tvos"
      }
    }
  }
  feature {
    name: "linker_param_file"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
      }
    }
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
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
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-c"
        flag: "%{source_file}"
        expand_if_all_available: "source_file"
      }
    }
  }
  feature {
    name: "compiler_output_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-S"
        expand_if_all_available: "output_assembly_file"
      }
      flag_group {
        flag: "-E"
        expand_if_all_available: "output_preprocess_file"
      }
      flag_group {
        flag: "-o"
        flag: "%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "objcopy_embed_flags"
    flag_set {
      action: "objcopy_embed_data"
      flag_group {
        flag: "-I"
        flag: "binary"
      }
    }
    enabled: true
  }
  action_config {
    config_name: "strip"
    action_name: "strip"
    tool {
      tool_path: "/usr/bin/strip"
    }
    flag_set {
      flag_group {
        flag: "-S"
        flag: "-o"
        flag: "%{output_file}"
      }
      flag_group {
        flag: "%{stripopts}"
        iterate_over: "stripopts"
      }
      flag_group {
        flag: "%{input_file}"
      }
    }
  }
  action_config {
    config_name: "c-compile"
    action_name: "c-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "unfiltered_cxx_flags"
  }
  action_config {
    config_name: "c++-compile"
    action_name: "c++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "unfiltered_cxx_flags"
  }
  action_config {
    config_name: "linkstamp-compile"
    action_name: "linkstamp-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-module-compile"
    action_name: "c++-module-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "unfiltered_cxx_flags"
  }
  action_config {
    config_name: "c++-header-parsing"
    action_name: "c++-header-parsing"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "unfiltered_cxx_flags"
  }
  action_config {
    config_name: "objc-compile"
    action_name: "objc-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "arm64"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "objc_actions"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
  }
  action_config {
    config_name: "objc++-compile"
    action_name: "objc++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "arm64"
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
  }
  action_config {
    config_name: "assemble"
    action_name: "assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "include_system_dirs"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "unfiltered_cxx_flags"
  }
  action_config {
    config_name: "preprocess-assemble"
    action_name: "preprocess-assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "unfiltered_cxx_flags"
  }
  action_config {
    config_name: "objc-archive"
    action_name: "objc-archive"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-filelist"
        flag: "%{obj_list_path}"
        flag: "-arch_only"
        flag: "arm64"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{archive_path}"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-executable"
    action_name: "objc-executable"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "arm64"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "objc++-executable"
    action_name: "objc++-executable"
    tool {
      tool_path: "wrapped_clang_pp"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
      flag_group {
        flag: "-arch"
        flag: "arm64"
      }
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "c++-link-executable"
    action_name: "c++-link-executable"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "symbol_counts"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "force_pic_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
    implies: "cpp_linker_flags"
  }
  action_config {
    config_name: "c++-link-dynamic-library"
    action_name: "c++-link-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
    implies: "cpp_linker_flags"
  }
  action_config {
    config_name: "c++-link-nodeps-dynamic-library"
    action_name: "c++-link-nodeps-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
    implies: "cpp_linker_flags"
  }
  action_config {
    config_name: "c++-link-static-library"
    action_name: "c++-link-static-library"
    tool {
      tool_path: "wrapped_ar"
      execution_requirement: "requires-darwin"
    }
    implies: "runtime_root_flags"
    implies: "archiver_flags"
    implies: "input_param_flags"
    implies: "linker_param_file"
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-fully-link"
    action_name: "objc-fully-link"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-arch_only"
        flag: "arm64"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{fully_linked_archive_path}"
      }
      flag_group {
        flag: "%{objc_library_exec_paths}"
        iterate_over: "objc_library_exec_paths"
      }
      flag_group {
        flag: "%{cc_library_exec_paths}"
        iterate_over: "cc_library_exec_paths"
      }
      flag_group {
        flag: "%{imported_library_exec_paths}"
        iterate_over: "imported_library_exec_paths"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objcopy_embed_data"
    action_name: "objcopy_embed_data"
    tool {
      tool_path: "/usr/bin/objcopy"
    }
    enabled: true
  }
  cc_target_os: "apple"
}
toolchain {
  toolchain_identifier: "ios_arm64"
  host_system_name: "x86_64-apple-macosx"
  target_system_name: "arm64-apple-ios"
  target_cpu: "ios_arm64"
  target_libc: "ios"
  compiler: "compiler"
  abi_version: "local"
  abi_libc_version: "local"
  tool_path {
    name: "ar"
    path: "wrapped_ar"
  }
  tool_path {
    name: "compat-ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "cpp"
    path: "/usr/bin/cpp"
  }
  tool_path {
    name: "dwp"
    path: "/usr/bin/dwp"
  }
  tool_path {
    name: "gcc"
    path: "cc_wrapper.sh"
  }
  tool_path {
    name: "gcov"
    path: "/usr/bin/gcov"
  }
  tool_path {
    name: "ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "nm"
    path: "/usr/bin/nm"
  }
  tool_path {
    name: "objcopy"
    path: "/usr/bin/objcopy"
  }
  tool_path {
    name: "objdump"
    path: "/usr/bin/objdump"
  }
  tool_path {
    name: "strip"
    path: "/usr/bin/strip"
  }
  make_variable {
    name: "STACK_FRAME_UNLIMITED"
    value: "-Wframe-larger-than=100000000 -Wno-vla"
  }
  %{cxx_builtin_include_directory}
  builtin_sysroot: ""
  feature {
    name: "fastbuild"
  }
  feature {
    name: "no_legacy_features"
  }
  feature {
    name: "opt"
  }
  feature {
    name: "dbg"
  }
  feature {
    name: "link_libc++"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-lc++"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    enabled: true
  }
  feature {
    name: "compile_all_modules"
  }
  feature {
    name: "exclude_private_headers_in_module_maps"
  }
  feature {
    name: "has_configured_linker_path"
  }
  feature {
    name: "only_doth_headers_in_module_maps"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-fstack-protector"
        flag: "-fcolor-diagnostics"
        flag: "-Wall"
        flag: "-Wthread-safety"
        flag: "-Wself-assign"
        flag: "-fno-omit-frame-pointer"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-O0"
        flag: "-DDEBUG"
      }
      with_feature {
        feature: "fastbuild"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g0"
        flag: "-O2"
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-DNDEBUG"
        flag: "-ffunction-sections"
        flag: "-fdata-sections"
      }
      with_feature {
        feature: "opt"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g"
      }
      with_feature {
        feature: "dbg"
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
        flag: "-std=c++11"
      }
    }
    enabled: true
  }
  feature {
    name: "debug_prefix_map_pwd_is_dot"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "DEBUG_PREFIX_MAP_PWD=."
      }
    }
  }
  feature {
    name: "generate_dsym_file"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-g"
      }
    }
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "DSYM_HINT_LINKED_BINARY=%{linked_binary}"
        flag: "DSYM_HINT_DSYM_PATH=%{dsym_path}"
      }
    }
  }
  feature {
    name: "contains_objc_source"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fobjc-link-runtime"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "objc_actions"
    implies: "objc-compile"
    implies: "objc++-compile"
    implies: "objc-fully-link"
    implies: "objc-archive"
    implies: "objc-executable"
    implies: "objc++-executable"
    implies: "assemble"
    implies: "preprocess-assemble"
    implies: "c-compile"
    implies: "c++-compile"
    implies: "c++-link-static-library"
    implies: "c++-link-dynamic-library"
    implies: "c++-link-nodeps-dynamic-library"
    implies: "c++-link-executable"
  }
  feature {
    name: "strip_debug_symbols"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-Wl,-S"
        expand_if_all_available: "strip_debug_symbols"
      }
    }
  }
  feature {
    name: "symbol_counts"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,--print-symbol-counts=%{symbol_counts_output}"
        expand_if_all_available: "symbol_counts_output"
      }
    }
  }
  feature {
    name: "shared_flag"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-shared"
      }
    }
  }
  feature {
    name: "kernel_extension"
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
        flag: "-o"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "archiver_flags"
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "rcS"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "runtime_root_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-Wl,-rpath,@loader_path/%{runtime_library_search_directories}"
        iterate_over: "runtime_library_search_directories"
        expand_if_all_available: "runtime_library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_flags}"
        iterate_over: "runtime_root_flags"
        expand_if_all_available: "runtime_root_flags"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_entries}"
        iterate_over: "runtime_root_entries"
        expand_if_all_available: "runtime_root_entries"
      }
    }
  }
  feature {
    name: "input_param_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-L%{library_search_directories}"
        iterate_over: "library_search_directories"
        expand_if_all_available: "library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
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
        flag: "-Wl,-force_load,%{whole_archive_linker_params}"
        iterate_over: "whole_archive_linker_params"
        expand_if_all_available: "whole_archive_linker_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{linker_input_params}"
        iterate_over: "linker_input_params"
        expand_if_all_available: "linker_input_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag_group {
          flag: "-Wl,--start-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.object_files}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.object_files}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          iterate_over: "libraries_to_link.object_files"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag: "-Wl,--end-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
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
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "static_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "dynamic_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l:%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l:%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "versioned_dynamic_library"
          }
        }
        iterate_over: "libraries_to_link"
        expand_if_all_available: "libraries_to_link"
      }
    }
  }
  feature {
    name: "force_pic_flags"
    flag_set {
      action: "c++-link-executable"
      flag_group {
        flag: "-Wl,-pie"
        expand_if_all_available: "force_pic"
      }
    }
  }
  feature {
    name: "pch"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-include"
        flag: "%{pch_file}"
      }
    }
  }
  feature {
    name: "module_maps"
  }
  feature {
    name: "use_objc_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-name=%{module_name}"
        flag: "-iquote"
        flag: "%{module_maps_dir}"
        flag: "-fmodules-cache-path=%{modules_cache_path}"
      }
    }
  }
  feature {
    name: "no_enable_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-maps"
      }
    }
    requires {
      feature: "use_objc_modules"
    }
  }
  feature {
    name: "apply_default_warnings"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-Wshorten-64-to-32"
        flag: "-Wbool-conversion"
        flag: "-Wconstant-conversion"
        flag: "-Wduplicate-method-match"
        flag: "-Wempty-body"
        flag: "-Wenum-conversion"
        flag: "-Wint-conversion"
        flag: "-Wunreachable-code"
        flag: "-Wmismatched-return-types"
        flag: "-Wundeclared-selector"
        flag: "-Wuninitialized"
        flag: "-Wunused-function"
        flag: "-Wunused-variable"
      }
    }
  }
  feature {
    name: "includes"
    flag_set {
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "clif-match"
      flag_group {
        flag: "-include"
        flag: "%{includes}"
        iterate_over: "includes"
        expand_if_all_available: "includes"
      }
    }
    enabled: true
  }
  feature {
    name: "include_paths"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "clif-match"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-iquote"
        flag: "%{quote_include_paths}"
        iterate_over: "quote_include_paths"
      }
      flag_group {
        flag: "-I%{include_paths}"
        iterate_over: "include_paths"
      }
      flag_group {
        flag: "-isystem"
        flag: "%{system_include_paths}"
        iterate_over: "system_include_paths"
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
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "linkstamp-compile"
      action: "clif-match"
      flag_group {
        flag: "--sysroot=%{sysroot}"
        expand_if_all_available: "sysroot"
      }
    }
  }
  feature {
    name: "dependency_file"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      flag_group {
        flag: "-MD"
        flag: "-MF"
        flag: "%{dependency_file}"
        expand_if_all_available: "dependency_file"
      }
    }
  }
  feature {
    name: "pic"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "preprocess-assemble"
      flag_group {
        flag: "-fPIC"
        expand_if_all_available: "pic"
      }
    }
  }
  feature {
    name: "per_object_debug_info"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-gsplit-dwarf"
        expand_if_all_available: "per_object_debug_info_file"
      }
    }
  }
  feature {
    name: "preprocessor_defines"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D%{preprocessor_defines}"
        iterate_over: "preprocessor_defines"
      }
    }
  }
  feature {
    name: "framework_paths"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-F%{framework_paths}"
        iterate_over: "framework_paths"
      }
    }
  }
  feature {
    name: "random_seed"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      flag_group {
        flag: "-frandom-seed=%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "fdo_instrument"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fprofile-generate=%{fdo_instrument_path}"
        flag: "-fno-data-sections"
        expand_if_all_available: "fdo_instrument_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "fdo_optimize"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fprofile-use=%{fdo_profile_path}"
        flag: "-Xclang-only=-Wno-profile-instr-unprofiled"
        flag: "-Xclang-only=-Wno-profile-instr-out-of-date"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "autofdo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fauto-profile=%{fdo_profile_path}"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "lipo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fripa"
      }
    }
    requires {
      feature: "autofdo"
    }
    requires {
      feature: "fdo_optimize"
    }
    requires {
      feature: "fdo_instrument"
    }
  }
  feature {
    name: "coverage"
  }
  feature {
    name: "llvm_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-instr-generate"
        flag: "-fcoverage-mapping"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-fprofile-instr-generate"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "gcc_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-arcs"
        flag: "-ftest-coverage"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "--coverage"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "apply_default_compiler_flags"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-DOS_IOS"
        flag: "-fno-autolink"
      }
    }
  }
  feature {
    name: "include_system_dirs"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-isysroot"
        flag: "%{sdk_dir}"
      }
    }
  }
  feature {
    name: "bitcode_embedded"
  }
  feature {
    name: "bitcode_embedded_markers"
  }
  feature {
    name: "objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fobjc-arc"
        expand_if_all_available: "objc_arc"
      }
    }
  }
  feature {
    name: "no_objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fno-objc-arc"
        expand_if_all_available: "no_objc_arc"
      }
    }
  }
  feature {
    name: "apple_env"
    env_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-archive"
      action: "objc-fully-link"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      action: "objc-executable"
      action: "objc++-executable"
      action: "linkstamp-compile"
      env_entry {
        key: "XCODE_VERSION_OVERRIDE"
        value: "%{xcode_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_VERSION_OVERRIDE"
        value: "%{apple_sdk_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_PLATFORM"
        value: "%{apple_sdk_platform_value}"
      }
    }
  }
  feature {
    name: "user_link_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "%{user_link_flags}"
        iterate_over: "user_link_flags"
        expand_if_all_available: "user_link_flags"
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
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-headerpad_max_install_names"
        flag: "-no-canonical-prefixes"
        flag: "-target"
        flag: "arm64-apple-ios"
      }
    }
    enabled: true
  }
  feature {
    name: "version_min"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-miphoneos-version-min=%{version_min}"
      }
    }
  }
  feature {
    name: "dead_strip"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-dead_strip"
        flag: "-no_dead_strip_inits_and_terms"
      }
    }
    requires {
      feature: "opt"
    }
  }
  feature {
    name: "cpp_linker_flags"
  }
  feature {
    name: "apply_implicit_frameworks"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-framework"
        flag: "Foundation"
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "link_cocoa"
  }
  feature {
    name: "apply_simulator_compiler_flags"
  }
  feature {
    name: "unfiltered_cxx_flags"
  }
  feature {
    name: "user_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "%{user_compile_flags}"
        iterate_over: "user_compile_flags"
        expand_if_all_available: "user_compile_flags"
      }
    }
  }
  feature {
    name: "unfiltered_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      flag_group {
        flag: "-no-canonical-prefixes"
        flag: "-Wno-builtin-macro-redefined"
        flag: "-D__DATE__=\"redacted\""
        flag: "-D__TIMESTAMP__=\"redacted\""
        flag: "-D__TIME__=\"redacted\""
        flag: "-target"
        flag: "arm64-apple-ios"
      }
    }
  }
  feature {
    name: "linker_param_file"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
      }
    }
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
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
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-c"
        flag: "%{source_file}"
        expand_if_all_available: "source_file"
      }
    }
  }
  feature {
    name: "compiler_output_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-S"
        expand_if_all_available: "output_assembly_file"
      }
      flag_group {
        flag: "-E"
        expand_if_all_available: "output_preprocess_file"
      }
      flag_group {
        flag: "-o"
        flag: "%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "objcopy_embed_flags"
    flag_set {
      action: "objcopy_embed_data"
      flag_group {
        flag: "-I"
        flag: "binary"
      }
    }
    enabled: true
  }
  action_config {
    config_name: "strip"
    action_name: "strip"
    tool {
      tool_path: "/usr/bin/strip"
    }
    flag_set {
      flag_group {
        flag: "-S"
        flag: "-o"
        flag: "%{output_file}"
      }
      flag_group {
        flag: "%{stripopts}"
        iterate_over: "stripopts"
      }
      flag_group {
        flag: "%{input_file}"
      }
    }
  }
  action_config {
    config_name: "c-compile"
    action_name: "c-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-compile"
    action_name: "c++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "linkstamp-compile"
    action_name: "linkstamp-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-module-compile"
    action_name: "c++-module-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-header-parsing"
    action_name: "c++-header-parsing"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-compile"
    action_name: "objc-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "arm64"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "objc_actions"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
  }
  action_config {
    config_name: "objc++-compile"
    action_name: "objc++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "arm64"
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
  }
  action_config {
    config_name: "assemble"
    action_name: "assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "include_system_dirs"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "preprocess-assemble"
    action_name: "preprocess-assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-archive"
    action_name: "objc-archive"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-filelist"
        flag: "%{obj_list_path}"
        flag: "-arch_only"
        flag: "arm64"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{archive_path}"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-executable"
    action_name: "objc-executable"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "arm64"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "objc++-executable"
    action_name: "objc++-executable"
    tool {
      tool_path: "wrapped_clang_pp"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
      flag_group {
        flag: "-arch"
        flag: "arm64"
      }
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "c++-link-executable"
    action_name: "c++-link-executable"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "symbol_counts"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "force_pic_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-dynamic-library"
    action_name: "c++-link-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-nodeps-dynamic-library"
    action_name: "c++-link-nodeps-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-static-library"
    action_name: "c++-link-static-library"
    tool {
      tool_path: "wrapped_ar"
      execution_requirement: "requires-darwin"
    }
    implies: "runtime_root_flags"
    implies: "archiver_flags"
    implies: "input_param_flags"
    implies: "linker_param_file"
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-fully-link"
    action_name: "objc-fully-link"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-arch_only"
        flag: "arm64"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{fully_linked_archive_path}"
      }
      flag_group {
        flag: "%{objc_library_exec_paths}"
        iterate_over: "objc_library_exec_paths"
      }
      flag_group {
        flag: "%{cc_library_exec_paths}"
        iterate_over: "cc_library_exec_paths"
      }
      flag_group {
        flag: "%{imported_library_exec_paths}"
        iterate_over: "imported_library_exec_paths"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objcopy_embed_data"
    action_name: "objcopy_embed_data"
    tool {
      tool_path: "/usr/bin/objcopy"
    }
    enabled: true
  }
  cc_target_os: "apple"
}
toolchain {
  toolchain_identifier: "ios_arm64e"
  host_system_name: "x86_64-apple-macosx"
  target_system_name: "arm64e-apple-ios"
  target_cpu: "ios_arm64e"
  target_libc: "ios"
  compiler: "compiler"
  abi_version: "local"
  abi_libc_version: "local"
  tool_path {
    name: "ar"
    path: "wrapped_ar"
  }
  tool_path {
    name: "compat-ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "cpp"
    path: "/usr/bin/cpp"
  }
  tool_path {
    name: "dwp"
    path: "/usr/bin/dwp"
  }
  tool_path {
    name: "gcc"
    path: "cc_wrapper.sh"
  }
  tool_path {
    name: "gcov"
    path: "/usr/bin/gcov"
  }
  tool_path {
    name: "ld"
    path: "/usr/bin/ld"
  }
  tool_path {
    name: "nm"
    path: "/usr/bin/nm"
  }
  tool_path {
    name: "objcopy"
    path: "/usr/bin/objcopy"
  }
  tool_path {
    name: "objdump"
    path: "/usr/bin/objdump"
  }
  tool_path {
    name: "strip"
    path: "/usr/bin/strip"
  }
  make_variable {
    name: "STACK_FRAME_UNLIMITED"
    value: "-Wframe-larger-than=100000000 -Wno-vla"
  }
  %{cxx_builtin_include_directory}
  builtin_sysroot: ""
  feature {
    name: "fastbuild"
  }
  feature {
    name: "no_legacy_features"
  }
  feature {
    name: "opt"
  }
  feature {
    name: "dbg"
  }
  feature {
    name: "link_libc++"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-lc++"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    enabled: true
  }
  feature {
    name: "compile_all_modules"
  }
  feature {
    name: "exclude_private_headers_in_module_maps"
  }
  feature {
    name: "has_configured_linker_path"
  }
  feature {
    name: "only_doth_headers_in_module_maps"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-fstack-protector"
        flag: "-fcolor-diagnostics"
        flag: "-Wall"
        flag: "-Wthread-safety"
        flag: "-Wself-assign"
        flag: "-fno-omit-frame-pointer"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-O0"
        flag: "-DDEBUG"
      }
      with_feature {
        feature: "fastbuild"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g0"
        flag: "-O2"
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-DNDEBUG"
        flag: "-ffunction-sections"
        flag: "-fdata-sections"
        flag: "-DNS_BLOCK_ASSERTIONS=1"
      }
      with_feature {
        feature: "opt"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g"
      }
      with_feature {
        feature: "dbg"
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
        flag: "-std=c++11"
      }
    }
    enabled: true
  }
  feature {
    name: "debug_prefix_map_pwd_is_dot"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "DEBUG_PREFIX_MAP_PWD=."
      }
    }
  }
  feature {
    name: "generate_dsym_file"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-g"
      }
    }
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "DSYM_HINT_LINKED_BINARY=%{linked_binary}"
        flag: "DSYM_HINT_DSYM_PATH=%{dsym_path}"
      }
    }
  }
  feature {
    name: "contains_objc_source"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fobjc-link-runtime"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "objc_actions"
    implies: "objc-compile"
    implies: "objc++-compile"
    implies: "objc-fully-link"
    implies: "objc-archive"
    implies: "objc-executable"
    implies: "objc++-executable"
    implies: "assemble"
    implies: "preprocess-assemble"
    implies: "c-compile"
    implies: "c++-compile"
    implies: "c++-link-static-library"
    implies: "c++-link-dynamic-library"
    implies: "c++-link-nodeps-dynamic-library"
    implies: "c++-link-executable"
  }
  feature {
    name: "strip_debug_symbols"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-Wl,-S"
        expand_if_all_available: "strip_debug_symbols"
      }
    }
  }
  feature {
    name: "symbol_counts"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,--print-symbol-counts=%{symbol_counts_output}"
        expand_if_all_available: "symbol_counts_output"
      }
    }
  }
  feature {
    name: "shared_flag"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-shared"
      }
    }
  }
  feature {
    name: "kernel_extension"
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
        flag: "-o"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "archiver_flags"
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "rcS"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "runtime_root_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-Wl,-rpath,@loader_path/%{runtime_library_search_directories}"
        iterate_over: "runtime_library_search_directories"
        expand_if_all_available: "runtime_library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_flags}"
        iterate_over: "runtime_root_flags"
        expand_if_all_available: "runtime_root_flags"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_entries}"
        iterate_over: "runtime_root_entries"
        expand_if_all_available: "runtime_root_entries"
      }
    }
  }
  feature {
    name: "input_param_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-L%{library_search_directories}"
        iterate_over: "library_search_directories"
        expand_if_all_available: "library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
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
        flag: "-Wl,-force_load,%{whole_archive_linker_params}"
        iterate_over: "whole_archive_linker_params"
        expand_if_all_available: "whole_archive_linker_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{linker_input_params}"
        iterate_over: "linker_input_params"
        expand_if_all_available: "linker_input_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag_group {
          flag: "-Wl,--start-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.object_files}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.object_files}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          iterate_over: "libraries_to_link.object_files"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag: "-Wl,--end-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
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
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "static_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "dynamic_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l:%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l:%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "versioned_dynamic_library"
          }
        }
        iterate_over: "libraries_to_link"
        expand_if_all_available: "libraries_to_link"
      }
    }
  }
  feature {
    name: "force_pic_flags"
    flag_set {
      action: "c++-link-executable"
      flag_group {
        flag: "-Wl,-pie"
        expand_if_all_available: "force_pic"
      }
    }
  }
  feature {
    name: "pch"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-include"
        flag: "%{pch_file}"
      }
    }
  }
  feature {
    name: "module_maps"
  }
  feature {
    name: "use_objc_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-name=%{module_name}"
        flag: "-iquote"
        flag: "%{module_maps_dir}"
        flag: "-fmodules-cache-path=%{modules_cache_path}"
      }
    }
  }
  feature {
    name: "no_enable_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-maps"
      }
    }
    requires {
      feature: "use_objc_modules"
    }
  }
  feature {
    name: "apply_default_warnings"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-Wshorten-64-to-32"
        flag: "-Wbool-conversion"
        flag: "-Wconstant-conversion"
        flag: "-Wduplicate-method-match"
        flag: "-Wempty-body"
        flag: "-Wenum-conversion"
        flag: "-Wint-conversion"
        flag: "-Wunreachable-code"
        flag: "-Wmismatched-return-types"
        flag: "-Wundeclared-selector"
        flag: "-Wuninitialized"
        flag: "-Wunused-function"
        flag: "-Wunused-variable"
      }
    }
  }
  feature {
    name: "includes"
    flag_set {
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "clif-match"
      flag_group {
        flag: "-include"
        flag: "%{includes}"
        iterate_over: "includes"
        expand_if_all_available: "includes"
      }
    }
    enabled: true
  }
  feature {
    name: "include_paths"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "clif-match"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-iquote"
        flag: "%{quote_include_paths}"
        iterate_over: "quote_include_paths"
      }
      flag_group {
        flag: "-I%{include_paths}"
        iterate_over: "include_paths"
      }
      flag_group {
        flag: "-isystem"
        flag: "%{system_include_paths}"
        iterate_over: "system_include_paths"
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
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "linkstamp-compile"
      action: "clif-match"
      flag_group {
        flag: "--sysroot=%{sysroot}"
        expand_if_all_available: "sysroot"
      }
    }
  }
  feature {
    name: "dependency_file"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      flag_group {
        flag: "-MD"
        flag: "-MF"
        flag: "%{dependency_file}"
        expand_if_all_available: "dependency_file"
      }
    }
  }
  feature {
    name: "pic"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "preprocess-assemble"
      flag_group {
        flag: "-fPIC"
        expand_if_all_available: "pic"
      }
    }
  }
  feature {
    name: "per_object_debug_info"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-gsplit-dwarf"
        expand_if_all_available: "per_object_debug_info_file"
      }
    }
  }
  feature {
    name: "preprocessor_defines"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D%{preprocessor_defines}"
        iterate_over: "preprocessor_defines"
      }
    }
  }
  feature {
    name: "framework_paths"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-F%{framework_paths}"
        iterate_over: "framework_paths"
      }
    }
  }
  feature {
    name: "random_seed"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      flag_group {
        flag: "-frandom-seed=%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "fdo_instrument"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fprofile-generate=%{fdo_instrument_path}"
        flag: "-fno-data-sections"
        expand_if_all_available: "fdo_instrument_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "fdo_optimize"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fprofile-use=%{fdo_profile_path}"
        flag: "-Xclang-only=-Wno-profile-instr-unprofiled"
        flag: "-Xclang-only=-Wno-profile-instr-out-of-date"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "autofdo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fauto-profile=%{fdo_profile_path}"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "lipo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fripa"
      }
    }
    requires {
      feature: "autofdo"
    }
    requires {
      feature: "fdo_optimize"
    }
    requires {
      feature: "fdo_instrument"
    }
  }
  feature {
    name: "coverage"
  }
  feature {
    name: "llvm_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-instr-generate"
        flag: "-fcoverage-mapping"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-fprofile-instr-generate"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "gcc_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-arcs"
        flag: "-ftest-coverage"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "--coverage"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "apply_default_compiler_flags"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-DOS_IOS"
        flag: "-fno-autolink"
      }
    }
  }
  feature {
    name: "include_system_dirs"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-isysroot"
        flag: "%{sdk_dir}"
      }
    }
  }
  feature {
    name: "bitcode_embedded"
  }
  feature {
    name: "bitcode_embedded_markers"
  }
  feature {
    name: "objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fobjc-arc"
        expand_if_all_available: "objc_arc"
      }
    }
  }
  feature {
    name: "no_objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fno-objc-arc"
        expand_if_all_available: "no_objc_arc"
      }
    }
  }
  feature {
    name: "apple_env"
    env_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-archive"
      action: "objc-fully-link"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      action: "objc-executable"
      action: "objc++-executable"
      action: "linkstamp-compile"
      env_entry {
        key: "XCODE_VERSION_OVERRIDE"
        value: "%{xcode_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_VERSION_OVERRIDE"
        value: "%{apple_sdk_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_PLATFORM"
        value: "%{apple_sdk_platform_value}"
      }
    }
  }
  feature {
    name: "user_link_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "%{user_link_flags}"
        iterate_over: "user_link_flags"
        expand_if_all_available: "user_link_flags"
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
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-headerpad_max_install_names"
        flag: "-no-canonical-prefixes"
        flag: "-target"
        flag: "arm64e-apple-ios"
      }
    }
    enabled: true
  }
  feature {
    name: "version_min"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-miphoneos-version-min=%{version_min}"
      }
    }
  }
  feature {
    name: "dead_strip"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-dead_strip"
        flag: "-no_dead_strip_inits_and_terms"
      }
    }
    requires {
      feature: "opt"
    }
  }
  feature {
    name: "cpp_linker_flags"
  }
  feature {
    name: "apply_implicit_frameworks"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-framework"
        flag: "Foundation"
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "link_cocoa"
  }
  feature {
    name: "apply_simulator_compiler_flags"
  }
  feature {
    name: "unfiltered_cxx_flags"
  }
  feature {
    name: "user_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "%{user_compile_flags}"
        iterate_over: "user_compile_flags"
        expand_if_all_available: "user_compile_flags"
      }
    }
  }
  feature {
    name: "unfiltered_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      flag_group {
        flag: "-no-canonical-prefixes"
        flag: "-Wno-builtin-macro-redefined"
        flag: "-D__DATE__=\"redacted\""
        flag: "-D__TIMESTAMP__=\"redacted\""
        flag: "-D__TIME__=\"redacted\""
        flag: "-target"
        flag: "arm64e-apple-ios"
      }
    }
  }
  feature {
    name: "linker_param_file"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
      }
    }
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
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
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-c"
        flag: "%{source_file}"
        expand_if_all_available: "source_file"
      }
    }
  }
  feature {
    name: "compiler_output_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-S"
        expand_if_all_available: "output_assembly_file"
      }
      flag_group {
        flag: "-E"
        expand_if_all_available: "output_preprocess_file"
      }
      flag_group {
        flag: "-o"
        flag: "%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "objcopy_embed_flags"
    flag_set {
      action: "objcopy_embed_data"
      flag_group {
        flag: "-I"
        flag: "binary"
      }
    }
    enabled: true
  }
  action_config {
    config_name: "strip"
    action_name: "strip"
    tool {
      tool_path: "/usr/bin/strip"
    }
    flag_set {
      flag_group {
        flag: "-S"
        flag: "-o"
        flag: "%{output_file}"
      }
      flag_group {
        flag: "%{stripopts}"
        iterate_over: "stripopts"
      }
      flag_group {
        flag: "%{input_file}"
      }
    }
  }
  action_config {
    config_name: "c-compile"
    action_name: "c-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-compile"
    action_name: "c++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "linkstamp-compile"
    action_name: "linkstamp-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-module-compile"
    action_name: "c++-module-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-header-parsing"
    action_name: "c++-header-parsing"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-compile"
    action_name: "objc-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "arm64e"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "objc_actions"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
  }
  action_config {
    config_name: "objc++-compile"
    action_name: "objc++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "arm64e"
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
  }
  action_config {
    config_name: "assemble"
    action_name: "assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "include_system_dirs"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "preprocess-assemble"
    action_name: "preprocess-assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-archive"
    action_name: "objc-archive"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-filelist"
        flag: "%{obj_list_path}"
        flag: "-arch_only"
        flag: "arm64e"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{archive_path}"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-executable"
    action_name: "objc-executable"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "arm64e"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "objc++-executable"
    action_name: "objc++-executable"
    tool {
      tool_path: "wrapped_clang_pp"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
      flag_group {
        flag: "-arch"
        flag: "arm64e"
      }
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "c++-link-executable"
    action_name: "c++-link-executable"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "symbol_counts"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "force_pic_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-dynamic-library"
    action_name: "c++-link-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-nodeps-dynamic-library"
    action_name: "c++-link-nodeps-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-static-library"
    action_name: "c++-link-static-library"
    tool {
      tool_path: "wrapped_ar"
      execution_requirement: "requires-darwin"
    }
    implies: "runtime_root_flags"
    implies: "archiver_flags"
    implies: "input_param_flags"
    implies: "linker_param_file"
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-fully-link"
    action_name: "objc-fully-link"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-arch_only"
        flag: "arm64e"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{fully_linked_archive_path}"
      }
      flag_group {
        flag: "%{objc_library_exec_paths}"
        iterate_over: "objc_library_exec_paths"
      }
      flag_group {
        flag: "%{cc_library_exec_paths}"
        iterate_over: "cc_library_exec_paths"
      }
      flag_group {
        flag: "%{imported_library_exec_paths}"
        iterate_over: "imported_library_exec_paths"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objcopy_embed_data"
    action_name: "objcopy_embed_data"
    tool {
      tool_path: "/usr/bin/objcopy"
    }
    enabled: true
  }
  cc_target_os: "apple"
}
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
  make_variable {
    name: "STACK_FRAME_UNLIMITED"
    value: "-Wframe-larger-than=100000000 -Wno-vla"
  }
  %{cxx_builtin_include_directory}
  builtin_sysroot: ""
  feature {
    name: "fastbuild"
  }
  feature {
    name: "no_legacy_features"
  }
  feature {
    name: "opt"
  }
  feature {
    name: "dbg"
  }
  feature {
    name: "link_libc++"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-lc++"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    enabled: true
  }
  feature {
    name: "compile_all_modules"
  }
  feature {
    name: "exclude_private_headers_in_module_maps"
  }
  feature {
    name: "has_configured_linker_path"
  }
  feature {
    name: "only_doth_headers_in_module_maps"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-fstack-protector"
        flag: "-fcolor-diagnostics"
        flag: "-Wall"
        flag: "-Wthread-safety"
        flag: "-Wself-assign"
        flag: "-fno-omit-frame-pointer"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-O0"
        flag: "-DDEBUG"
      }
      with_feature {
        feature: "fastbuild"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g0"
        flag: "-O2"
        flag: "-D_FORTIFY_SOURCE=1"
        flag: "-DNDEBUG"
        flag: "-ffunction-sections"
        flag: "-fdata-sections"
        flag: "-DNS_BLOCK_ASSERTIONS=1"
      }
      with_feature {
        feature: "opt"
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
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-g"
      }
      with_feature {
        feature: "dbg"
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
        flag: "-std=c++11"
      }
    }
    enabled: true
  }
  feature {
    name: "debug_prefix_map_pwd_is_dot"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "DEBUG_PREFIX_MAP_PWD=."
      }
    }
  }
  feature {
    name: "generate_dsym_file"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-g"
      }
    }
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "DSYM_HINT_LINKED_BINARY=%{linked_binary}"
        flag: "DSYM_HINT_DSYM_PATH=%{dsym_path}"
      }
    }
  }
  feature {
    name: "contains_objc_source"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fobjc-link-runtime"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "objc_actions"
    implies: "objc-compile"
    implies: "objc++-compile"
    implies: "objc-fully-link"
    implies: "objc-archive"
    implies: "objc-executable"
    implies: "objc++-executable"
    implies: "assemble"
    implies: "preprocess-assemble"
    implies: "c-compile"
    implies: "c++-compile"
    implies: "c++-link-static-library"
    implies: "c++-link-dynamic-library"
    implies: "c++-link-nodeps-dynamic-library"
    implies: "c++-link-executable"
  }
  feature {
    name: "strip_debug_symbols"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-Wl,-S"
        expand_if_all_available: "strip_debug_symbols"
      }
    }
  }
  feature {
    name: "symbol_counts"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,--print-symbol-counts=%{symbol_counts_output}"
        expand_if_all_available: "symbol_counts_output"
      }
    }
  }
  feature {
    name: "shared_flag"
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-shared"
      }
    }
  }
  feature {
    name: "kernel_extension"
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
        flag: "-o"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "archiver_flags"
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "rcS"
        flag: "%{output_execpath}"
        expand_if_all_available: "output_execpath"
      }
    }
  }
  feature {
    name: "runtime_root_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-Wl,-rpath,@loader_path/%{runtime_library_search_directories}"
        iterate_over: "runtime_library_search_directories"
        expand_if_all_available: "runtime_library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_flags}"
        iterate_over: "runtime_root_flags"
        expand_if_all_available: "runtime_root_flags"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{runtime_root_entries}"
        iterate_over: "runtime_root_entries"
        expand_if_all_available: "runtime_root_entries"
      }
    }
  }
  feature {
    name: "input_param_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "-L%{library_search_directories}"
        iterate_over: "library_search_directories"
        expand_if_all_available: "library_search_directories"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
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
        flag: "-Wl,-force_load,%{whole_archive_linker_params}"
        iterate_over: "whole_archive_linker_params"
        expand_if_all_available: "whole_archive_linker_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag: "%{linker_input_params}"
        iterate_over: "linker_input_params"
        expand_if_all_available: "linker_input_params"
      }
    }
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      flag_group {
        flag_group {
          flag: "-Wl,--start-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.object_files}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.object_files}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          iterate_over: "libraries_to_link.object_files"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag: "-Wl,--end-lib"
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file_group"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "object_file"
          }
        }
        flag_group {
          flag_group {
            flag: "%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
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
            flag: "-Wl,-force_load,%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "static_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "dynamic_library"
          }
        }
        flag_group {
          flag_group {
            flag: "-l:%{libraries_to_link.name}"
            expand_if_false: "libraries_to_link.is_whole_archive"
          }
          flag_group {
            flag: "-Wl,-force_load,-l:%{libraries_to_link.name}"
            expand_if_true: "libraries_to_link.is_whole_archive"
          }
          expand_if_equal {
            variable: "libraries_to_link.type"
            value: "versioned_dynamic_library"
          }
        }
        iterate_over: "libraries_to_link"
        expand_if_all_available: "libraries_to_link"
      }
    }
  }
  feature {
    name: "force_pic_flags"
    flag_set {
      action: "c++-link-executable"
      flag_group {
        flag: "-Wl,-pie"
        expand_if_all_available: "force_pic"
      }
    }
  }
  feature {
    name: "pch"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-include"
        flag: "%{pch_file}"
      }
    }
  }
  feature {
    name: "module_maps"
  }
  feature {
    name: "use_objc_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-name=%{module_name}"
        flag: "-iquote"
        flag: "%{module_maps_dir}"
        flag: "-fmodules-cache-path=%{modules_cache_path}"
      }
    }
  }
  feature {
    name: "no_enable_modules"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fmodule-maps"
      }
    }
    requires {
      feature: "use_objc_modules"
    }
  }
  feature {
    name: "apply_default_warnings"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-Wshorten-64-to-32"
        flag: "-Wbool-conversion"
        flag: "-Wconstant-conversion"
        flag: "-Wduplicate-method-match"
        flag: "-Wempty-body"
        flag: "-Wenum-conversion"
        flag: "-Wint-conversion"
        flag: "-Wunreachable-code"
        flag: "-Wmismatched-return-types"
        flag: "-Wundeclared-selector"
        flag: "-Wuninitialized"
        flag: "-Wunused-function"
        flag: "-Wunused-variable"
      }
    }
  }
  feature {
    name: "includes"
    flag_set {
      action: "preprocess-assemble"
      action: "linkstamp-compile"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "clif-match"
      flag_group {
        flag: "-include"
        flag: "%{includes}"
        iterate_over: "includes"
        expand_if_all_available: "includes"
      }
    }
    enabled: true
  }
  feature {
    name: "include_paths"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "clif-match"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-iquote"
        flag: "%{quote_include_paths}"
        iterate_over: "quote_include_paths"
      }
      flag_group {
        flag: "-I%{include_paths}"
        iterate_over: "include_paths"
      }
      flag_group {
        flag: "-isystem"
        flag: "%{system_include_paths}"
        iterate_over: "system_include_paths"
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
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "linkstamp-compile"
      action: "clif-match"
      flag_group {
        flag: "--sysroot=%{sysroot}"
        expand_if_all_available: "sysroot"
      }
    }
  }
  feature {
    name: "dependency_file"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      action: "c++-header-parsing"
      flag_group {
        flag: "-MD"
        flag: "-MF"
        flag: "%{dependency_file}"
        expand_if_all_available: "dependency_file"
      }
    }
  }
  feature {
    name: "pic"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "preprocess-assemble"
      flag_group {
        flag: "-fPIC"
        expand_if_all_available: "pic"
      }
    }
  }
  feature {
    name: "per_object_debug_info"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-gsplit-dwarf"
        expand_if_all_available: "per_object_debug_info_file"
      }
    }
  }
  feature {
    name: "preprocessor_defines"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-D%{preprocessor_defines}"
        iterate_over: "preprocessor_defines"
      }
    }
  }
  feature {
    name: "framework_paths"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-F%{framework_paths}"
        iterate_over: "framework_paths"
      }
    }
  }
  feature {
    name: "random_seed"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-codegen"
      action: "c++-module-compile"
      flag_group {
        flag: "-frandom-seed=%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "fdo_instrument"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "-fprofile-generate=%{fdo_instrument_path}"
        flag: "-fno-data-sections"
        expand_if_all_available: "fdo_instrument_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "fdo_optimize"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fprofile-use=%{fdo_profile_path}"
        flag: "-Xclang-only=-Wno-profile-instr-unprofiled"
        flag: "-Xclang-only=-Wno-profile-instr-out-of-date"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "autofdo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fauto-profile=%{fdo_profile_path}"
        flag: "-fprofile-correction"
        expand_if_all_available: "fdo_profile_path"
      }
    }
    provides: "profile"
  }
  feature {
    name: "lipo"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      flag_group {
        flag: "-fripa"
      }
    }
    requires {
      feature: "autofdo"
    }
    requires {
      feature: "fdo_optimize"
    }
    requires {
      feature: "fdo_instrument"
    }
  }
  feature {
    name: "coverage"
  }
  feature {
    name: "llvm_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-instr-generate"
        flag: "-fcoverage-mapping"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-fprofile-instr-generate"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "gcc_coverage_map_format"
    flag_set {
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fprofile-arcs"
        flag: "-ftest-coverage"
        flag: "-g"
      }
    }
    flag_set {
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-executable"
      flag_group {
        flag: "--coverage"
      }
    }
    requires {
      feature: "coverage"
    }
  }
  feature {
    name: "apply_default_compiler_flags"
    flag_set {
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-DOS_IOS"
        flag: "-fno-autolink"
      }
    }
  }
  feature {
    name: "include_system_dirs"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-executable"
      action: "objc++-executable"
      action: "assemble"
      action: "preprocess-assemble"
      flag_group {
        flag: "-isysroot"
        flag: "%{sdk_dir}"
      }
    }
  }
  feature {
    name: "bitcode_embedded"
  }
  feature {
    name: "bitcode_embedded_markers"
  }
  feature {
    name: "objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fobjc-arc"
        expand_if_all_available: "objc_arc"
      }
    }
  }
  feature {
    name: "no_objc_arc"
    flag_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-fno-objc-arc"
        expand_if_all_available: "no_objc_arc"
      }
    }
  }
  feature {
    name: "apple_env"
    env_set {
      action: "c-compile"
      action: "c++-compile"
      action: "c++-module-compile"
      action: "c++-header-parsing"
      action: "assemble"
      action: "preprocess-assemble"
      action: "objc-compile"
      action: "objc++-compile"
      action: "objc-archive"
      action: "objc-fully-link"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "c++-link-static-library"
      action: "objc-executable"
      action: "objc++-executable"
      action: "linkstamp-compile"
      env_entry {
        key: "XCODE_VERSION_OVERRIDE"
        value: "%{xcode_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_VERSION_OVERRIDE"
        value: "%{apple_sdk_version_override_value}"
      }
      env_entry {
        key: "APPLE_SDK_PLATFORM"
        value: "%{apple_sdk_platform_value}"
      }
    }
  }
  feature {
    name: "user_link_flags"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "%{user_link_flags}"
        iterate_over: "user_link_flags"
        expand_if_all_available: "user_link_flags"
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
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-headerpad_max_install_names"
        flag: "-no-canonical-prefixes"
      }
    }
    enabled: true
  }
  feature {
    name: "version_min"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-m<platform_for_version_min>-version-min=%{version_min}"
      }
    }
  }
  feature {
    name: "dead_strip"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-dead_strip"
        flag: "-no_dead_strip_inits_and_terms"
      }
    }
    requires {
      feature: "opt"
    }
  }
  feature {
    name: "cpp_linker_flags"
  }
  feature {
    name: "apply_implicit_frameworks"
    flag_set {
      action: "objc-executable"
      action: "objc++-executable"
      flag_group {
        flag: "-framework"
        flag: "Foundation"
        flag: "-framework"
        flag: "UIKit"
      }
    }
  }
  feature {
    name: "link_cocoa"
  }
  feature {
    name: "apply_simulator_compiler_flags"
  }
  feature {
    name: "unfiltered_cxx_flags"
  }
  feature {
    name: "user_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "%{user_compile_flags}"
        iterate_over: "user_compile_flags"
        expand_if_all_available: "user_compile_flags"
      }
    }
  }
  feature {
    name: "unfiltered_compile_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "linkstamp-compile"
      flag_group {
        flag: "-no-canonical-prefixes"
        flag: "-Wno-builtin-macro-redefined"
        flag: "-D__DATE__=\"redacted\""
        flag: "-D__TIMESTAMP__=\"redacted\""
        flag: "-D__TIME__=\"redacted\""
      }
    }
  }
  feature {
    name: "linker_param_file"
    flag_set {
      action: "c++-link-executable"
      action: "c++-link-dynamic-library"
      action: "c++-link-nodeps-dynamic-library"
      flag_group {
        flag: "-Wl,@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
      }
    }
    flag_set {
      action: "c++-link-static-library"
      flag_group {
        flag: "@%{linker_param_file}"
        expand_if_all_available: "linker_param_file"
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
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-c"
        flag: "%{source_file}"
        expand_if_all_available: "source_file"
      }
    }
  }
  feature {
    name: "compiler_output_flags"
    flag_set {
      action: "assemble"
      action: "preprocess-assemble"
      action: "c-compile"
      action: "c++-compile"
      action: "linkstamp-compile"
      action: "c++-header-parsing"
      action: "c++-module-compile"
      action: "c++-module-codegen"
      action: "objc-compile"
      action: "objc++-compile"
      flag_group {
        flag: "-S"
        expand_if_all_available: "output_assembly_file"
      }
      flag_group {
        flag: "-E"
        expand_if_all_available: "output_preprocess_file"
      }
      flag_group {
        flag: "-o"
        flag: "%{output_file}"
        expand_if_all_available: "output_file"
      }
    }
  }
  feature {
    name: "supports_pic"
    enabled: true
  }
  feature {
    name: "objcopy_embed_flags"
    flag_set {
      action: "objcopy_embed_data"
      flag_group {
        flag: "-I"
        flag: "binary"
      }
    }
    enabled: true
  }
  action_config {
    config_name: "strip"
    action_name: "strip"
    tool {
      tool_path: "/usr/bin/strip"
    }
    flag_set {
      flag_group {
        flag: "-S"
        flag: "-o"
        flag: "%{output_file}"
      }
      flag_group {
        flag: "%{stripopts}"
        iterate_over: "stripopts"
      }
      flag_group {
        flag: "%{input_file}"
      }
    }
  }
  action_config {
    config_name: "c-compile"
    action_name: "c-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-compile"
    action_name: "c++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "linkstamp-compile"
    action_name: "linkstamp-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-module-compile"
    action_name: "c++-module-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "c++-header-parsing"
    action_name: "c++-header-parsing"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-compile"
    action_name: "objc-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "<architecture>"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "objc_actions"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
  }
  action_config {
    config_name: "objc++-compile"
    action_name: "objc++-compile"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "<architecture>"
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
    }
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
    implies: "apply_default_compiler_flags"
    implies: "apply_default_warnings"
    implies: "framework_paths"
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
  }
  action_config {
    config_name: "assemble"
    action_name: "assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "include_system_dirs"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "preprocess-assemble"
    action_name: "preprocess-assemble"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    implies: "preprocessor_defines"
    implies: "include_system_dirs"
    implies: "version_min"
    implies: "objc_arc"
    implies: "no_objc_arc"
    implies: "apple_env"
    implies: "user_compile_flags"
    implies: "sysroot"
    implies: "unfiltered_compile_flags"
    implies: "compiler_input_flags"
    implies: "compiler_output_flags"
  }
  action_config {
    config_name: "objc-archive"
    action_name: "objc-archive"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-filelist"
        flag: "%{obj_list_path}"
        flag: "-arch_only"
        flag: "<architecture>"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{archive_path}"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-executable"
    action_name: "objc-executable"
    tool {
      tool_path: "wrapped_clang"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      with_feature {
        not_feature: "kernel_extension"
      }
    }
    flag_set {
      flag_group {
        flag: "-arch"
        flag: "<architecture>"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "objc++-executable"
    action_name: "objc++-executable"
    tool {
      tool_path: "wrapped_clang_pp"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-stdlib=libc++"
        flag: "-std=gnu++11"
      }
      flag_group {
        flag: "-arch"
        flag: "<architecture>"
      }
      flag_group {
        flag: "-Xlinker"
        flag: "-objc_abi_version"
        flag: "-Xlinker"
        flag: "2"
        flag: "-Xlinker"
        flag: "-rpath"
        flag: "-Xlinker"
        flag: "@executable_path/Frameworks"
        flag: "-fobjc-link-runtime"
        flag: "-ObjC"
      }
      flag_group {
        flag: "-framework"
        flag: "%{framework_names}"
        iterate_over: "framework_names"
      }
      flag_group {
        flag: "-weak_framework"
        flag: "%{weak_framework_names}"
        iterate_over: "weak_framework_names"
      }
      flag_group {
        flag: "-l%{library_names}"
        iterate_over: "library_names"
      }
      flag_group {
        flag: "-filelist"
        flag: "%{filelist}"
      }
      flag_group {
        flag: "-o"
        flag: "%{linked_binary}"
      }
      flag_group {
        flag: "-force_load"
        flag: "%{force_load_exec_paths}"
        iterate_over: "force_load_exec_paths"
      }
      flag_group {
        flag: "%{dep_linkopts}"
        iterate_over: "dep_linkopts"
      }
      flag_group {
        flag: "-Wl,%{attr_linkopts}"
        iterate_over: "attr_linkopts"
      }
    }
    implies: "include_system_dirs"
    implies: "framework_paths"
    implies: "version_min"
    implies: "strip_debug_symbols"
    implies: "apple_env"
    implies: "apply_implicit_frameworks"
  }
  action_config {
    config_name: "c++-link-executable"
    action_name: "c++-link-executable"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "symbol_counts"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "force_pic_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-dynamic-library"
    action_name: "c++-link-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-nodeps-dynamic-library"
    action_name: "c++-link-nodeps-dynamic-library"
    tool {
      tool_path: "cc_wrapper.sh"
      execution_requirement: "requires-darwin"
    }
    implies: "contains_objc_source"
    implies: "has_configured_linker_path"
    implies: "symbol_counts"
    implies: "shared_flag"
    implies: "linkstamps"
    implies: "output_execpath_flags"
    implies: "runtime_root_flags"
    implies: "input_param_flags"
    implies: "strip_debug_symbols"
    implies: "linker_param_file"
    implies: "version_min"
    implies: "apple_env"
    implies: "sysroot"
  }
  action_config {
    config_name: "c++-link-static-library"
    action_name: "c++-link-static-library"
    tool {
      tool_path: "wrapped_ar"
      execution_requirement: "requires-darwin"
    }
    implies: "runtime_root_flags"
    implies: "archiver_flags"
    implies: "input_param_flags"
    implies: "linker_param_file"
    implies: "apple_env"
  }
  action_config {
    config_name: "objc-fully-link"
    action_name: "objc-fully-link"
    tool {
      tool_path: "libtool"
      execution_requirement: "requires-darwin"
    }
    flag_set {
      flag_group {
        flag: "-no_warning_for_no_symbols"
        flag: "-static"
        flag: "-arch_only"
        flag: "<architecture>"
        flag: "-syslibroot"
        flag: "%{sdk_dir}"
        flag: "-o"
        flag: "%{fully_linked_archive_path}"
      }
      flag_group {
        flag: "%{objc_library_exec_paths}"
        iterate_over: "objc_library_exec_paths"
      }
      flag_group {
        flag: "%{cc_library_exec_paths}"
        iterate_over: "cc_library_exec_paths"
      }
      flag_group {
        flag: "%{imported_library_exec_paths}"
        iterate_over: "imported_library_exec_paths"
      }
    }
    implies: "apple_env"
  }
  action_config {
    config_name: "objcopy_embed_data"
    action_name: "objcopy_embed_data"
    tool {
      tool_path: "/bin/false"
    }
    enabled: true
  }
  cc_target_os: "apple"
}

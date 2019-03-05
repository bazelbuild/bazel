// Copyright 2015 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.packages.util;

import com.google.common.base.Joiner;
import com.google.common.base.Predicate;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.rules.cpp.CppActionNames;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.protobuf.TextFormat;
import java.io.IOException;

/**
 * Creates mock BUILD files required for the C/C++ rules.
 */
public abstract class MockCcSupport {

  /** Filter to remove implicit crosstool artifact and module map inputs of C/C++ rules. */
  public static final Predicate<Artifact> CC_ARTIFACT_FILTER =
      new Predicate<Artifact>() {
        @Override
        public boolean apply(Artifact artifact) {
          String basename = artifact.getExecPath().getBaseName();
          String pathString = artifact.getExecPathString();
          return !pathString.startsWith("third_party/crosstool/")
              && !pathString.startsWith("tools/cpp/link_dynamic_library")
              && !pathString.startsWith("tools/cpp/build_interface_so")
              && !(pathString.contains("/internal/_middlemen") && basename.contains("crosstool"))
              && !pathString.startsWith("_bin/build_interface_so")
              && !pathString.endsWith(".cppmap")
              && !pathString.startsWith("tools/cpp/grep-includes");
        }
      };

  /** This feature will prevent bazel from patching the crosstool. */
  public static final String NO_LEGACY_FEATURES_FEATURE = "feature { name: 'no_legacy_features' }";

  public static final String STARLARK_NO_LEGACY_FEATURES_FEATURE =
      "[feature(name = 'no_legacy_features')]";

  public static final String DYNAMIC_LINKING_MODE_FEATURE =
      "feature { name: '" + CppRuleClasses.DYNAMIC_LINKING_MODE + "'}";

  public static final String STARLARK_DYNAMIC_LINKING_MODE_FEATURE =
      "[feature(name = '" + CppRuleClasses.DYNAMIC_LINKING_MODE + "')]";

  public static final String SUPPORTS_DYNAMIC_LINKER_FEATURE =
      "feature { name: '" + CppRuleClasses.SUPPORTS_DYNAMIC_LINKER + "' enabled: true}";

  public static final String STARLARK_SUPPORTS_DYNAMIC_LINKER_FEATURE =
      "[feature(name = '" + CppRuleClasses.SUPPORTS_DYNAMIC_LINKER + "', enabled = True)]";

  public static final String SUPPORTS_INTERFACE_SHARED_LIBRARIES_FEATURE =
      "feature { name: '" + CppRuleClasses.SUPPORTS_INTERFACE_SHARED_LIBRARIES + "' enabled: true}";

  public static final String STARLARK_SUPPORTS_INTERFACE_SHARED_LIBRARIES_FEATURE =
      Joiner.on("\n")
          .join(
              "[feature(",
              "        name = '" + CppRuleClasses.SUPPORTS_INTERFACE_SHARED_LIBRARIES + "',",
              "'        enabled = True)]");

  /** Feature expected by the C++ rules when pic build is requested */
  public static final String PIC_FEATURE =
      ""
          + "feature {"
          + "  name: 'pic'"
          + "  enabled: true"
          + "  flag_set {"
          + "    action: 'assemble'"
          + "    action: 'preprocess-assemble'"
          + "    action: 'linkstamp-compile'"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-module-codegen'"
          + "    action: 'c++-module-compile'"
          + "    flag_group {"
          + "      expand_if_all_available: 'pic'"
          + "      flag: '-fPIC'"
          + "    }"
          + "  }"
          + "}";

  public static final String STARLARK_PIC_FEATURE =
      Joiner.on("\n")
          .join(
              "[feature(",
              "        name = 'pic',",
              "        enabled = True,",
              "        flag_sets = [",
              "            flag_set(",
              "                actions = [",
              "                    'assemble',",
              "                    'preprocess-assemble',",
              "                    'linkstamp-compile',",
              "                    'c-compile',",
              "                    'c++-compile',",
              "                    'c++-module-codegen',",
              "                    'c++-module-compile',",
              "                ],",
              "                flag_groups = [",
              "                    flag_group(",
              "                        expand_if_available = 'pic',",
              "                        flags = ['-fPIC'],",
              "                    )",
              "                ],",
              "            ),",
              "        ],",
              "    )]");

  /** A feature configuration snippet useful for testing header processing. */
  public static final String PARSE_HEADERS_FEATURE_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'parse_headers'"
          + "  flag_set {"
          + "    action: 'c++-header-parsing'"
          + "    flag_group {"
          + "      flag: '<c++-header-parsing>'"
          + "    }"
          + "  }"
          + "}";

  public static final String STARLARK_PARSE_HEADERS_FEATURE_CONFIGURATION =
      Joiner.on("\n")
          .join(
              "[feature(",
              "        name = 'parse_headers',",
              "        flag_sets = [",
              "            flag_set(",
              "                actions = ['c++-header-parsing'],",
              "                flag_groups = [",
              "                    flag_group(",
              "                        flags = ['<c++-header-parsing>'],",
              "                    ),",
              "                ],",
              "            ),",
              "        ],",
              "    )]");

  /** A feature configuration snippet useful for testing the layering check. */
  public static final String LAYERING_CHECK_FEATURE_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'layering_check'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-module-compile'"
          + "    flag_group {"
          + "      iterate_over: 'dependent_module_map_files'"
          + "      flag: 'dependent_module_map_file:%{dependent_module_map_files}'"
          + "    }"
          + "  }"
          + "}";

  public static final String STARLARK_LAYERING_CHECK_FEATURE_CONFIGURATION =
      Joiner.on("\n")
          .join(
              "[feature(",
              "        name = 'layering_check',",
              "        flag_sets = [",
              "            flag_set(",
              "                actions = [",
              "                    'c-compile',",
              "                    'c++-compile',",
              "                    'c++-header-parsing',",
              "                    'c++-module-compile',",
              "                ],",
              "                flag_groups = [",
              "                    flag_group(",
              "                        iterate_over = 'dependent_module_map_files',",
              "                        flags = [",
              "                            'dependent_module_map_file:"
                  + "%{dependent_module_map_files}',",
              "                        ],",
              "                    ),",
              "                ],",
              "            ),",
              "        ],",
              "    )]");

  /** A feature configuration snippet useful for testing header modules. */
  public static final String HEADER_MODULES_FEATURE_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'header_modules'"
          + "  implies: 'use_header_modules'"
          + "  implies: 'header_module_compile'"
          + "}"
          + "feature {"
          + "  name: 'header_module_compile'"
          + "  enabled: true"
          + "  implies: 'module_maps'"
          + "  flag_set {"
          + "    action: 'c++-module-compile'"
          + "    flag_group {"
          + "      flag: '--woohoo_modules'"
          + "    }"
          + "  }"
          + "  flag_set {"
          + "    action: 'c++-module-codegen'"
          + "    flag_group {"
          + "      flag: '--this_is_modules_codegen'"
          + "    }"
          + "  }"
          + "}"
          + "feature {"
          + "  name: 'header_module_codegen'"
          + "  implies: 'header_modules'"
          + "}"
          + "feature {"
          + "  name: 'module_maps'"
          + "  enabled: true"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-module-compile'"
          + "    flag_group {"
          + "      flag: 'module_name:%{module_name}'"
          + "      flag: 'module_map_file:%{module_map_file}'"
          + "    }"
          + "  }"
          + "}"
          + "feature {"
          + "  name: 'use_header_modules'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-modules-compile'"
          + "    flag_group {"
          + "      iterate_over: 'module_files'"
          + "      flag: 'module_file:%{module_files}'"
          + "    }"
          + "  }"
          + "}";

  public static final String STARLARK_HEADER_MODULES_FEATURE_CONFIGURATION =
      Joiner.on("\n")
          .join(
              "[feature(",
              "        name = 'header_modules',",
              "        implies = ['use_header_modules', 'header_module_compile'],",
              "    ),",
              "    feature(",
              "        name = 'header_module_compile',",
              "        enabled = True,",
              "        implies = ['module_maps'],",
              "        flag_sets = [",
              "            flag_set (",
              "                actions = ['c++-module-compile'],",
              "                flag_groups = [",
              "                    flag_group(",
              "                        flags = ['--woohoo_modules'],",
              "                    ),",
              "                ],",
              "            ),",
              "            flag_set(",
              "                actions = ['c++-module-codegen'],",
              "                flag_groups = [",
              "                    flag_group(",
              "                        flags = ['--this_is_modules_codegen'],",
              "                    ),",
              "                ],",
              "            ),",
              "        ],",
              "    ),",
              "    feature(",
              "        name = 'header_module_codegen',",
              "        implies = ['header_modules'],",
              "    ),",
              "    feature(",
              "        name = 'module_maps',",
              "        enabled: True,",
              "        flag_sets = [",
              "            flag_set (",
              "                actions = ['c-compile', 'c++-compile',",
              "                           'c++-header-parsing', 'c++-module-compile'],",
              "                flag_groups = [",
              "                    flag_group(",
              "                        flags = [",
              "                            'module_name:%{module_name}',",
              "                            'module_map_file:%{module_map_file}',",
              "                        ],",
              "                    ),",
              "                ],",
              "            ),",
              "        ],",
              "    ),",
              "    feature(",
              "        name = 'use_header_modules',",
              "        flag_sets = [",
              "            flag_set (",
              "                actions = ['c-compile', 'c++-compile',",
              "                           'c++-header-parsing', 'c++-module-compile'],",
              "                flag_groups = [",
              "                    flag_group (",
              "                        iterate_over = 'module_files',",
              "                        flags = ['module_file:%{module_files}'],",
              "                    ),",
              "                ],",
              "            ),",
              "        ],",
              "    )]");

  public static final String MODULE_MAP_HOME_CWD_FEATURE =
      ""
          + "feature {"
          + "  name: 'module_map_home_cwd'"
          + "  enabled: true"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-module-compile'"
          + "    action: 'preprocess-assemble'"
          + "    flag_group {"
          + "      flag: '<flag>'"
          + "    }"
          + "  }"
          + "}";

  public static final String STARLARK_MODULE_MAP_HOME_CWD_FEATURE =
      Joiner.on("\n")
          .join(
              "[feature(",
              "        name = 'module_map_home_cwd',",
              "        enabled = True,",
              "        flag_sets = [",
              "            flag_set(",
              "                actions = [",
              "                    'c-compile', 'c++-compile',",
              "                    'c++-header-parsing', 'c++-module-compile',",
              "                    'preprocess-assemble'",
              "                ],",
              "                flag_groups = [",
              "                    flag_group(",
              "                        flags = ['<flag>'],",
              "                    ),",
              "                ],",
              "            ),",
              "        ],",
              "    ),]");

  /** A feature configuration snippet useful for testing environment variables. */
  public static final String ENV_VAR_FEATURE_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'env_feature'"
          + "  implies: 'static_env_feature'"
          + "  implies: 'module_maps'"
          + "}"
          + "feature {"
          + "  name: 'static_env_feature'"
          + "  env_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-module-compile'"
          + "    env_entry {"
          + "      key: 'cat'"
          + "      value: 'meow'"
          + "    }"
          + "  }"
          + "}"
          + "feature {"
          + "  name: 'module_maps'"
          + "  enabled: true"
          + "  env_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-module-compile'"
          + "    env_entry {"
          + "      key: 'module'"
          + "      value: 'module_name:%{module_name}'"
          + "    }"
          + "  }"
          + "}";

  public static final String STARLARK_ENV_VAR_FEATURE_CONFIGURATION =
      Joiner.on("\n")
          .join(
              "[feature(",
              "        name = 'env_feature',",
              "        implies = ['static_env_feature', 'module_maps'],",
              "    ),",
              "    feature(",
              "        name = 'static_env_feature',",
              "        env_sets = [",
              "            env_set(",
              "                actions = ['c-compile', 'c++-compile',",
              "                           'c++-header-parsing', 'c++-module-compile',",
              "                ],",
              "                env_entries = [",
              "                    env_entry(",
              "                        key = 'cat',",
              "                        value = 'meow',",
              "                    ),",
              "                ],",
              "            ),",
              "        ],",
              "    ),",
              "    feature(",
              "        name = 'module_maps',",
              "        enabled = True,",
              "        env_sets = [",
              "            env_set(",
              "                actions = ['c-compile', 'c++-compile',",
              "                           'c++-header-parsing', 'c++-module-compile',",
              "                ],",
              "                env_entries = [",
              "                    env_entry(",
              "                        key = 'module',",
              "                        value = 'module_name:%{module_name}',",
              "                    ),",
              "                ],",
              "            ),",
              "        ],",
              "    )]");

  public static final String HOST_AND_NONHOST_CONFIGURATION =
      ""
          + "feature { "
          + "  name: 'host'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    flag_group {"
          + "      flag: '-host'"
          + "    }"
          + "  }"
          + "}"
          + "feature { "
          + "  name: 'nonhost'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    flag_group {"
          + "      flag: '-nonhost'"
          + "    }"
          + "  }"
          + "}";

  public static final String STARLARK_HOST_AND_NONHOST_CONFIGURATION =
      Joiner.on("\n")
          .join(
              "[feature(",
              "        name = 'host'",
              "        flag_sets = [",
              "            flag_set(",
              "                actions = ['c-compile', 'c-compile'],",
              "                flag_groups = [flag_group(flags = ['-host'])],",
              "            ),",
              "        ],",
              "    ),",
              "    feature(",
              "        name = 'nonhost',",
              "        flag_sets = [",
              "            flag_set(",
              "                actions = ['c-compile','c-compile'],",
              "                flag_groups = [flag_group(flags = ['-nonhost'])],",
              "            ),",
              "        ],",
              "    )]");

  public static final String THIN_LTO_CONFIGURATION =
      ""
          + "feature { "
          + "  name: 'thin_lto'"
          + "  requires { feature: 'nonhost' }"
          + "  flag_set {"
          + "    expand_if_all_available: 'thinlto_param_file'"
          + "    action: 'c++-link-executable'"
          + "    action: 'c++-link-dynamic-library'"
          + "    action: 'c++-link-nodeps-dynamic-library'"
          + "    action: 'c++-link-static-library'"
          + "    flag_group {"
          + "      flag: 'thinlto_param_file=%{thinlto_param_file}'"
          + "    }"
          + "  }"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    flag_group {"
          + "      flag: '-flto=thin'"
          + "    }"
          + "    flag_group {"
          + "      expand_if_all_available: 'lto_indexing_bitcode_file'"
          + "      flag: 'lto_indexing_bitcode=%{lto_indexing_bitcode_file}'"
          + "    }"
          + "  }"
          + "  flag_set {"
          + "    action: 'lto-indexing'"
          + "    flag_group {"
          + "      flag: 'param_file=%{thinlto_indexing_param_file}'"
          + "      flag: 'prefix_replace=%{thinlto_prefix_replace}'"
          + "    }"
          + "    flag_group {"
          + "      expand_if_all_available: 'thinlto_object_suffix_replace'"
          + "      flag: 'object_suffix_replace=%{thinlto_object_suffix_replace}'"
          + "    }"
          + "    flag_group {"
          + "      expand_if_all_available: 'thinlto_merged_object_file'"
          + "      flag: 'thinlto_merged_object_file=%{thinlto_merged_object_file}'"
          + "    }"
          + "  }"
          + "  flag_set {"
          + "    action: 'lto-backend'"
          + "    flag_group {"
          + "      flag: 'thinlto_index=%{thinlto_index}'"
          + "      flag: 'thinlto_output_object_file=%{thinlto_output_object_file}'"
          + "      flag: 'thinlto_input_bitcode_file=%{thinlto_input_bitcode_file}'"
          + "    }"
          + "  }"
          + "}";

  public static final String STARLARK_THIN_LTO_CONFIGURATION =
      Joiner.on("\n")
          .join(
              "[feature(",
              "        name = 'thin_lto',",
              "        flag_sets = [",
              "            flag_set(",
              "                actions = [",
              "                    'c++-link-executable',",
              "                    'c++-link-dynamic-library',",
              "                    'c++-link-nodeps-dynamic-library',",
              "                    'c++-link-static-library'",
              "                ],",
              "                flag_groups = [",
              "                    flag_group(",
              "                        flags = ['thinlto_param_file=%{thinlto_param_file}'],",
              "                    ),",
              "                ],",
              "           ),",
              "           flag_set(",
              "               actions = ['c-compile', 'c++-compile'],",
              "               flag_groups = [",
              "                   flag_group(flags = ['-flto=thin']),",
              "                   flag_group(",
              "                       flags = [",
              "                           'lto_indexing_bitcode=%{lto_indexing_bitcode_file}'],",
              "                       expand_if_available = 'lto_indexing_bitcode_file',",
              "                   ),",
              "               ],",
              "           ),",
              "           flag_set(",
              "               actions = ['lto-indexing'],",
              "               flag_groups = [",
              "                   flag_group(",
              "                       flags = [",
              "                           'param_file=%{thinlto_indexing_param_file}',",
              "                           'prefix_replace=%{thinlto_prefix_replace}',",
              "                       ],",
              "                   ),",
              "                   flag_group(",
              "                       flags = [",
              "                           'object_suffix_replace=%{thinlto_object_suffix_replace}'",
              "                       ],",
              "                       expand_if_available = 'thinlto_object_suffix_replace',",
              "                   ),",
              "                   flag_group(",
              "                       flags = [",
              "                           'thinlto_merged_object_file="
                  + "%{thinlto_merged_object_file}',",
              "                       ],",
              "                       expand_if_available = 'thinlto_merged_object_file',",
              "                   ),",
              "               ],",
              "           ),",
              "           flag_set(",
              "               actions = ['lto-backend'],",
              "               flag_groups = [",
              "                   flag_group(",
              "                       flags = [",
              "                           'thinlto_index=%{thinlto_index}',",
              "                           'thinlto_output_object_file="
                  + "%{thinlto_output_object_file}',",
              "                           'thinlto_input_bitcode_file="
                  + "%{thinlto_input_bitcode_file}',",
              "                        ],",
              "                   ),",
              "               ],",
              "           ),",
              "        ],",
              "       requires = [feature_set(features = ['nonhost'])],",
              "   )]");

  public static final String THIN_LTO_LINKSTATIC_TESTS_USE_SHARED_NONLTO_BACKENDS_CONFIGURATION =
      "" + "feature {  name: 'thin_lto_linkstatic_tests_use_shared_nonlto_backends'}";

  public static final String
      STARLARK_THIN_LTO_LINKSTATIC_TESTS_USE_SHARED_NONLTO_BACKENDS_CONFIGURATION =
          "[feature(name = 'thin_lto_linkstatic_tests_use_shared_nonlto_backends')]";

  public static final String THIN_LTO_ALL_LINKSTATIC_USE_SHARED_NONLTO_BACKENDS_CONFIGURATION =
      "" + "feature {  name: 'thin_lto_all_linkstatic_use_shared_nonlto_backends'}";

  public static final String
      STARLARK_THIN_LTO_ALL_LINKSTATIC_USE_SHARED_NONLTO_BACKENDS_CONFIGURATION =
          "[feature(name = 'thin_lto_all_linkstatic_use_shared_nonlto_backends')]";

  public static final String ENABLE_AFDO_THINLTO_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'enable_afdo_thinlto'"
          + "  requires { feature: 'autofdo_implicit_thinlto' }"
          + "  implies: 'thin_lto'"
          + "}";

  public static final String STARLARK_ENABLE_AFDO_THINLTO_CONFIGURATION =
      Joiner.on("")
          .join(
              "[feature(",
              "        name = 'enable_fdo_thinlto',",
              "        requires = [feature_set(features = ['fdo_implicit_thinlto'])],",
              "        implies = ['thin_lto'],",
              "    )]");

  public static final String AUTOFDO_IMPLICIT_THINLTO_CONFIGURATION =
      "" + "feature {  name: 'autofdo_implicit_thinlto'}";

  public static final String STARLARK_AUTOFDO_IMPLICIT_THINLTO_CONFIGURATION =
      "[feature(name = 'autofdo_implicit_thinlto')]";

  public static final String ENABLE_FDO_THINLTO_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'enable_fdo_thinlto'"
          + "  requires { feature: 'fdo_implicit_thinlto' }"
          + "  implies: 'thin_lto'"
          + "}";

  public static final String STARLARK_ENABLE_FDO_THINLTO_CONFIGURATION =
      Joiner.on("\n")
          .join(
              "[feature(",
              "        name = 'enable_afdo_thinlto',",
              "        requires = [feature_set(features = ['autofdo_implicit_thinlto'])],",
              "        implies = ['thin_lto'],",
              "    )]");

  public static final String FDO_IMPLICIT_THINLTO_CONFIGURATION =
      "" + "feature {  name: 'fdo_implicit_thinlto'}";

  public static final String STARLARK_FDO_IMPLICIT_THINLTO_CONFIGURATION =
      "[feature(name = 'fdo_implicit_thinlto')]";

  public static final String ENABLE_XFDO_THINLTO_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'enable_xbinaryfdo_thinlto'"
          + "  requires { feature: 'xbinaryfdo_implicit_thinlto' }"
          + "  implies: 'thin_lto'"
          + "}";

  public static final String STARLARK_ENABLE_XFDO_THINLTO_CONFIGURATION =
      Joiner.on("\n")
          .join(
              "[feature(",
              "        name = 'enable_xbinaryfdo_thinlto',",
              "        requires = [feature_set(features = ['xbinaryfdo_implicit_thinlto'])],",
              "        implies = ['thin_lto'],",
              "    )]");

  public static final String XFDO_IMPLICIT_THINLTO_CONFIGURATION =
      "" + "feature {  name: 'xbinaryfdo_implicit_thinlto'}";

  public static final String STARLARK_XFDO_IMPLICIT_THINLTO_CONFIGURATION =
      "[feature(name = 'xbinaryfdo_implicit_thinlto')]";

  public static final String AUTO_FDO_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'autofdo'"
          + "  provides: 'profile'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'lto-backend'"
          + "    flag_group {"
          + "      expand_if_all_available: 'fdo_profile_path'"
          + "      flag: '-fauto-profile=%{fdo_profile_path}'"
          + "      flag: '-fprofile-correction'"
          + "    }"
          + "  }"
          + "}";

  public static final String STARLARK_AUTO_FDO_CONFIGURATION =
      Joiner.on("\n")
          .join(
              "[feature(",
              "        name = 'autofdo',",
              "        flag_sets = [",
              "            flag_set(",
              "                actions = [",
              "                    'c-compile',",
              "                    'c++-compile',",
              "                    'lto-backend',",
              "                ],",
              "                flag_groups = [",
              "                    flag_group(",
              "                        flags = [",
              "                            '-fauto-profile=%{fdo_profile_path}',",
              "                            '-fprofile-correction',",
              "                        ],",
              "                        expand_if_available = 'fdo_profile_path',",
              "                    ),",
              "                ],",
              "            ),",
              "        ],",
              "        provides = ['profile'],",
              "    )]");

  public static final String IS_CC_FAKE_BINARY_CONFIGURATION =
      "feature { name: 'is_cc_fake_binary' }";

  public static final String STARLARK_IS_CC_FAKE_BINARY_CONFIGURATION =
      "[feature(name = 'is_cc_fake_binary')]";

  public static final String XBINARY_FDO_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'xbinaryfdo'"
          + "  provides: 'profile'"
          + "  flag_set {"
          + "    with_feature { not_feature: 'is_cc_fake_binary' }"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'lto-backend'"
          + "    expand_if_all_available: 'fdo_profile_path'"
          + "    flag_group {"
          + "      flag: '-fauto-profile=%{fdo_profile_path}'"
          + "      flag: '-fprofile-correction'"
          + "    }"
          + "  }"
          + "}";

  public static final String STARLARK_XBINARY_FDO_CONFIGURATION =
      Joiner.on("\n")
          .join(
              "[feature(",
              "        name = 'xbinaryfdo',",
              "         flag_sets = [",
              "            flag_set(",
              "                actions = [",
              "                    'c-compile',",
              "                    'c++-compile',",
              "                    'lto-backend',",
              "                ],",
              "                flag_groups = [",
              "                    flag_group(",
              "                        flags = [",
              "                            '-fauto-profile=%{fdo_profile_path}',",
              "                            '-fprofile-correction',",
              "                        ],",
              "                    ),",
              "                ],",
              "                with_features = [",
              "                    with_feature_set(not_features = ['is_cc_fake_binary']),",
              "                ],",
              "            ),",
              "        ],",
              "        provides = ['profile'],",
              "    )]");

  public static final String FDO_OPTIMIZE_CONFIGURATION =
      ""
          + "feature {"
          + "  name: 'fdo_optimize'"
          + "  provides: 'profile'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    flag_group {"
          + "      expand_if_all_available: 'fdo_profile_path'"
          + "      flag: '-fprofile-use=%{fdo_profile_path}'"
          + "      flag: '-Xclang-only=-Wno-profile-instr-unprofiled'"
          + "      flag: '-Xclang-only=-Wno-profile-instr-out-of-date'"
          + "      flag: '-Xclang-only=-Wno-backend-plugin'"
          + "      flag: '-fprofile-correction'"
          + "    }"
          + "  }"
          + "}";

  public static final String STARLARK_FDO_OPTIMIZE_CONFIGURATION =
      Joiner.on("\n")
          .join(
              "[feature(",
              "        name = 'fdo_optimize',",
              "        flag_sets = [",
              "            flag_set(",
              "                actions = ['c-compile', 'c++-compile'],",
              "                flag_groups = [",
              "                    flag_group(",
              "                        flags = [",
              "                            '-fprofile-use=%{fdo_profile_path}',",
              "                            '-Xclang-only=-Wno-profile-instr-unprofiled',",
              "                            '-Xclang-only=-Wno-profile-instr-out-of-date',",
              "                            '-Xclang-only=-Wno-backend-plugin',",
              "                            '-fprofile-correction',",
              "                        ],",
              "                        expand_if_available = 'fdo_profile_path',",
              "                   ),",
              "                ],",
              "            ),",
              "        ],",
              "        provides = ['profile'],",
              "    )]");

  public static final String FDO_INSTRUMENT_CONFIGURATION =
      ""
          + "feature { "
          + "  name: 'fdo_instrument'"
          + "  provides: 'profile'"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-link-dynamic-library'"
          + "    action: 'c++-link-nodeps-dynamic-library'"
          + "    action: 'c++-link-executable'"
          + "    flag_group {"
          + "      flag: 'fdo_instrument_option'"
          + "      flag: 'path=%{fdo_instrument_path}'"
          + "    }"
          + "  }"
          + "}";

  public static final String STARLARK_FDO_INSTRUMENT_CONFIGURATION =
      Joiner.on("\n")
          .join(
              "[feature(",
              "        name = 'fdo_instrument',",
              "        flag_sets = [",
              "            flag_set(",
              "                actions = [",
              "                    'c-compile',",
              "                    'c++-compile',",
              "                    'c++-link-dynamic-library',",
              "                    'c++-link-nodeps-dynamic-library',",
              "                    'c++-link-executable',",
              "                ],",
              "                flag_groups = [",
              "                    flag_group(",
              "                        flags = ['fdo_instrument_option',",
              "                                 'path=%{fdo_instrument_path}'],",
              "                    ),",
              "                ],",
              "            ),",
              "        ],",
              "        provides = ['profile'],",
              "    )]");

  public static final String PER_OBJECT_DEBUG_INFO_CONFIGURATION =
      ""
          + "feature { "
          + "  name: 'per_object_debug_info'"
          + "  enabled: true"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'assemble'"
          + "    action: 'preprocess-assemble'"
          + "    action: 'c++-module-codegen'"
          + "    action: 'lto-backend'"
          + "    flag_group {"
          + "      expand_if_all_available: 'per_object_debug_info_file'"
          + "      flag: 'per_object_debug_info_option'"
          + "    }"
          + "  }"
          + "}";

  public static final String STARLARK_PER_OBJECT_DEBUG_INFO_CONFIGURATION =
      Joiner.on("\n")
          .join(
              "[feature(",
              "        name = 'per_object_debug_info',",
              "        enabled = True,",
              "        flag_sets = [",
              "            flag_set(",
              "                actions = [",
              "                    'c-compile',",
              "                    'c++-compile',",
              "                    'assemble',",
              "                    'preprocess-assemble',",
              "                    'c++-module-codegen',",
              "                    'lto-backend',",
              "                ],",
              "                flag_groups = [",
              "                    flag_group(",
              "                        flags = ['per_object_debug_info_option'],",
              "                        expand_if_available = 'per_object_debug_info_file',",
              "                    ),",
              "                ],",
              "            ),",
              "        ],",
              "    )]");

  public static final String COPY_DYNAMIC_LIBRARIES_TO_BINARY_CONFIGURATION =
      "" + "feature { name: 'copy_dynamic_libraries_to_binary' }";

  public static final String STARLARK_COPY_DYNAMIC_LIBRARIES_TO_BINARY_CONFIGURATION =
      "[feature(name = 'copy_dynamic_libraries_to_binary')]";

  public static final String SUPPORTS_START_END_LIB_FEATURE =
      "" + "feature { name: 'supports_start_end_lib' enabled: true }";

  public static final String STARLARK_SUPPORTS_START_END_LIB_FEATURE =
      "[feature(name = 'supports_start_end_lib', enabled = True)]";

  public static final String SUPPORTS_PIC_FEATURE =
      "" + "feature { name: 'supports_pic' enabled: true }";

  public static final String STARLARK_SUPPORTS_PIC_FEATURE =
      "[feature(name = 'supports_pic', enabled = True)]";

  public static final String TARGETS_WINDOWS_CONFIGURATION =
      ""
          + "feature {"
          + "   name: 'targets_windows'"
          + "   implies: 'copy_dynamic_libraries_to_binary'"
          + "   enabled: true"
          + "}";

  public static final String STARLARK_TARGETS_WINDOWS_CONFIGURATION =
      Joiner.on("\n")
          .join(
              "[feature(",
              "        name = 'targets_windows',",
              "        enabled = True,",
              "        implies = ['copy_dynamic_libraries_to_binary'],",
              "    )]");

  public static final String STATIC_LINK_TWEAKED_CONFIGURATION =
      ""
          + "artifact_name_pattern {"
          + "   category_name: 'static_library'"
          + "   prefix: 'lib'"
          + "   extension: '.lib'"
          + "}";

  public static final String STARLARK_STATIC_LINK_TWEAKED_CONFIGURATION =
      Joiner.on("\n")
          .join(
              "[artifact_name_pattern(",
              "            category_name = 'static_library',",
              "            prefix = 'lib',",
              "            extension = '.lib',",
              "        )]");

  public static final String STATIC_LINK_AS_DOT_A_CONFIGURATION =
      ""
          + "artifact_name_pattern {"
          + "   category_name: 'static_library'"
          + "   prefix: 'lib'"
          + "   extension: '.a'"
          + "}";

  public static final String STARLARK_STATIC_LINK_AS_DOT_A_CONFIGURATION =
      Joiner.on("\n")
          .join(
              "[artifact_name_pattern(",
              "        category_name = 'static_library',",
              "        prefix = 'lib',",
              "        extension = '.a',",
              "    )]");

  public static final String MODULE_MAPS_FEATURE =
      ""
          + "feature {"
          + "  name: 'module_maps'"
          + "  enabled: true"
          + "  flag_set {"
          + "    action: 'c-compile'"
          + "    action: 'c++-compile'"
          + "    action: 'c++-header-parsing'"
          + "    action: 'c++-module-compile'"
          + "    flag_group {"
          + "      flag: 'module_name:%{module_name}'"
          + "      flag: 'module_map_file:%{module_map_file}'"
          + "    }"
          + "  }"
          + "}";

  public static final String STARLARK_MODULE_MAPS_FEATURE =
      Joiner.on("\n")
          .join(
              "[feature(",
              "        name = 'module_maps',",
              "        enabled = True,",
              "        flag_sets = [",
              "            flag_set(",
              "                actions = [",
              "                    'c-compile',",
              "                    'c++-compile',",
              "                    'c++-header-parsing',",
              "                    'c++-module-compile',",
              "                ],",
              "                flag_groups = [",
              "                    flag_group(",
              "                       flags = [",
              "                           'module_name:%{module_name}',",
              "                           'module_map_file:%{module_map_file}',",
              "                       ],",
              "                    ),",
              "                ],",
              "            ),",
              "        ],",
              "    )]");

  public static final String EMPTY_COMPILE_ACTION_CONFIG =
      emptyActionConfigFor(CppActionNames.CPP_COMPILE);

  public static final String EMPTY_STARLARK_COMPILE_ACTION_CONFIG =
      emptyStarlarkActionConfigFor(CppActionNames.CPP_COMPILE);

  public static final String EMPTY_MODULE_CODEGEN_ACTION_CONFIG =
      emptyActionConfigFor(CppActionNames.CPP_MODULE_CODEGEN);

  public static final String EMPTY_STARLARK_MODULE_CODEGEN_ACTION_CONFIG =
      emptyStarlarkActionConfigFor(CppActionNames.CPP_MODULE_CODEGEN);

  public static final String EMPTY_MODULE_COMPILE_ACTION_CONFIG =
      emptyActionConfigFor(CppActionNames.CPP_MODULE_COMPILE);

  public static final String EMPTY_STARLARK_MODULE_COMPILE_ACTION_CONFIG =
      emptyStarlarkActionConfigFor(CppActionNames.CPP_MODULE_COMPILE);

  public static final String EMPTY_EXECUTABLE_ACTION_CONFIG =
      emptyActionConfigFor(LinkTargetType.EXECUTABLE.getActionName());

  public static final String EMPTY_STARLARK_EXECUTABLE_ACTION_CONFIG =
      emptyStarlarkActionConfigFor(LinkTargetType.EXECUTABLE.getActionName());

  public static final String EMPTY_DYNAMIC_LIBRARY_ACTION_CONFIG =
      emptyActionConfigFor(LinkTargetType.NODEPS_DYNAMIC_LIBRARY.getActionName());

  public static final String EMPTY_STARLARK_DYNAMIC_LIBRARY_ACTION_CONFIG =
      emptyStarlarkActionConfigFor(LinkTargetType.NODEPS_DYNAMIC_LIBRARY.getActionName());

  public static final String EMPTY_TRANSITIVE_DYNAMIC_LIBRARY_ACTION_CONFIG =
      emptyActionConfigFor(LinkTargetType.DYNAMIC_LIBRARY.getActionName());

  public static final String EMPTY_STARLARK_TRANSITIVE_DYNAMIC_LIBRARY_ACTION_CONFIG =
      emptyStarlarkActionConfigFor(LinkTargetType.DYNAMIC_LIBRARY.getActionName());

  public static final String EMPTY_STATIC_LIBRARY_ACTION_CONFIG =
      emptyActionConfigFor(LinkTargetType.STATIC_LIBRARY.getActionName());

  public static final String EMPTY_STARLARK_STATIC_LIBRARY_ACTION_CONFIG =
      emptyStarlarkActionConfigFor(LinkTargetType.STATIC_LIBRARY.getActionName());

  public static final String EMPTY_CLIF_MATCH_ACTION_CONFIG =
      emptyActionConfigFor(CppActionNames.CLIF_MATCH);

  public static final String EMPTY_STARLARK_CLIF_MATCH_ACTION_CONFIG =
      emptyStarlarkActionConfigFor(CppActionNames.CLIF_MATCH);

  public static final String EMPTY_STRIP_ACTION_CONFIG = emptyActionConfigFor(CppActionNames.STRIP);
  public static final String EMPTY_STARLARK_STRIP_ACTION_CONFIG =
      emptyStarlarkActionConfigFor(CppActionNames.STRIP);

  public static final String STATIC_LINK_CPP_RUNTIMES_FEATURE =
      "feature { name: 'static_link_cpp_runtimes' enabled: true }";

  public static final String STARLARK_STATIC_LINK_CPP_RUNTIMES_FEATURE =
      "[feature(name = 'static_link_cpp_runtimes', enabled = True)]";

  public static final String EMPTY_CROSSTOOL =
      "major_version: 'foo'\nminor_version:' foo'\n" + emptyToolchainForCpu("k8");

  public static final String EMPTY_CC_TOOLCHAIN_CONFIG =
      Joiner.on("\n")
          .join(
              "def _impl(ctx):",
              "    return cc_common.create_cc_toolchain_config_info(",
              "        ctx = ctx,",
              emptyStarlarkToolchainConfigForCpu("k8"),
              "    )",
              "cc_toolchain_config = rule(",
              "    implementation = _impl,",
              "    attrs = {",
              "        'cpu': attr.string(),",
              "        'compiler': attr.string(),",
              "    },",
              "    provides = [CcToolchainConfigInfo],",
              ")");

  public static String emptyToolchainForCpu(String cpu, String... append) {
    return Joiner.on("\n")
        .join(
            ImmutableList.builder()
                .add(
                    "toolchain {",
                    "  toolchain_identifier: 'mock-llvm-toolchain-" + cpu + "'",
                    "  host_system_name: 'mock-system-name-for-" + cpu + "'",
                    "  target_system_name: 'mock-target-system-name-for-" + cpu + "'",
                    "  target_cpu: '" + cpu + "'",
                    "  target_libc: 'mock-libc-for-" + cpu + "'",
                    "  compiler: 'mock-compiler-for-" + cpu + "'",
                    "  abi_version: 'mock-abi-version-for-" + cpu + "'",
                    "  abi_libc_version: 'mock-abi-libc-for-" + cpu + "'")
                .addAll(ImmutableList.copyOf(append))
                .add("}")
                .build());
  }

  public static String emptyStarlarkToolchainConfigForCpu(String cpu) {
    return Joiner.on("\n")
        .join(
            "        toolchain_identifier = 'mock-llvm-toolchain-" + cpu + "',",
            "        host_system_name = 'mock-system-name-for-" + cpu + "',",
            "        target_system_name = 'mock-target-system-name-for-" + cpu + "',",
            "        target_cpu = '" + cpu + "',",
            "        target_libc = 'mock-libc-for-" + cpu + "',",
            "        compiler = 'mock-compiler-for-" + cpu + "',",
            "        abi_version = 'mock-abi-version-for-" + cpu + "',",
            "        abi_libc_version = 'mock-abi-libc-for-" + cpu + "',");
  }

  /**
   * Creates action_config for {@code actionName} action using DUMMY_TOOL that doesn't imply any
   * features.
   */
  private static String emptyActionConfigFor(String actionName) {
    return String.format(
        "action_config {"
            + "  config_name: '%s'"
            + "  action_name: '%s'"
            + "  tool {"
            + "    tool_path: 'DUMMY_TOOL'"
            + "  }"
            + "}",
        actionName, actionName);
  }

  private static String emptyStarlarkActionConfigFor(String actionName) {
    return String.format(
        "[action_config("
            + "        action_name = '%s',"
            + "        tools = [tool(tool_path = 'DUMMY_TOOL')],"
            + "    )]",
        actionName);
  }

  /** Filter to remove implicit dependencies of C/C++ rules. */
  private final Predicate<Label> ccLabelFilter =
      new Predicate<Label>() {
        @Override
        public boolean apply(Label label) {
          return labelNameFilter().apply("//" + label.getPackageName());
        }
      };

  public static String mergeCrosstoolConfig(String original, CToolchain toolchain)
      throws TextFormat.ParseException {
    CrosstoolConfig.CrosstoolRelease.Builder builder =
        CrosstoolConfig.CrosstoolRelease.newBuilder();
    TextFormat.merge(original, builder);
    for (CToolchain.Builder toolchainBuilder : builder.getToolchainBuilderList()) {
      toolchainBuilder.mergeFrom(toolchain);
    }
    return TextFormat.printToString(builder.build());
  }

  public abstract Predicate<String> labelNameFilter();

  /**
   * Setup the support for building C/C++.
   */
  public abstract void setup(MockToolsConfig config) throws IOException;

  /**
   * Creates a crosstool package by merging {@code toolchain} with the default mock CROSSTOOL file.
   *
   * @param partialToolchain A string representation of a CToolchain protocol buffer; note that
   *        this is allowed to be a partial buffer (required fields may be omitted).
   */
  public void setupCrosstool(MockToolsConfig config, String... partialToolchain)
      throws IOException {
    setupCrosstool(config, /* appendToCurrentToolchain= */ true, partialToolchain);
  }

  public void setupCrosstool(
      MockToolsConfig config, boolean appendToCurrentToolchain, String... partialToolchain)
      throws IOException {
    String toolchainString = Joiner.on("\n").join(partialToolchain);
    String crosstoolFile;
    if (appendToCurrentToolchain) {
      CToolchain.Builder toolchainBuilder = CToolchain.newBuilder();
      TextFormat.merge(toolchainString, toolchainBuilder);
      crosstoolFile = mergeCrosstoolConfig(readCrosstoolFile(), toolchainBuilder.buildPartial());
    } else {
      crosstoolFile = readCrosstoolFile() + toolchainString;
    }
    createCrosstoolPackage(
        config,
        crosstoolFile);
  }

  protected void createCrosstoolPackage(
      MockToolsConfig config,
      String crosstoolFile)
      throws IOException {
    String crosstoolTop = getCrosstoolTopPathForConfig(config);
    if (config.isRealFileSystem()) {
      config.linkTools(getRealFilesystemTools(crosstoolTop));
    } else {
      new Crosstool(config, crosstoolTop)
          .setCrosstoolFile(getMockCrosstoolVersion(), crosstoolFile)
          .setSupportedArchs(getCrosstoolArchs())
          .setSupportsHeaderParsing(true)
          .write(/* disableCrosstool= */ false);
    }
  }

  protected String getCrosstoolTopPathForConfig(MockToolsConfig config) {
    if (config.isRealFileSystem()) {
      return getRealFilesystemCrosstoolTopPath();
    } else {
      return getMockCrosstoolPath();
    }
  }

  public abstract String getMockCrosstoolPath();

  public static PackageIdentifier getMockCrosstoolsTop() {
    try {
      return PackageIdentifier.create(
          RepositoryName.create(TestConstants.TOOLS_REPOSITORY),
          PathFragment.create(TestConstants.MOCK_CC_CROSSTOOL_PATH));
    } catch (LabelSyntaxException e) {
      Verify.verify(false);
      throw new AssertionError(e);
    }
  }

  protected String readCrosstoolFile() throws IOException {
    return ResourceLoader.readFromResources(
        "com/google/devtools/build/lib/analysis/mock/MOCK_CROSSTOOL");
  }

  public abstract String getMockCrosstoolVersion();

  public abstract Label getMockCrosstoolLabel();

  protected abstract ImmutableList<String> getCrosstoolArchs();

  protected abstract String[] getRealFilesystemTools(String crosstoolTop);

  protected abstract String getRealFilesystemCrosstoolTopPath();

  public final Predicate<Label> labelFilter() {
    return ccLabelFilter;
  }
}

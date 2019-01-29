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

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import java.io.IOException;

/**
 * Bazel implementation of {@link MockCcSupport}
 */
public final class BazelMockCcSupport extends MockCcSupport {
  public static final BazelMockCcSupport INSTANCE = new BazelMockCcSupport();

  private static final String MOCK_CROSSTOOL_PATH =
      "com/google/devtools/build/lib/analysis/mock/MOCK_CROSSTOOL";

  /** Filter to remove implicit dependencies of C/C++ rules. */
  private static final Predicate<String> CC_LABEL_NAME_FILTER =
      new Predicate<String>() {
        @Override
        public boolean apply(String label) {
          return !label.startsWith("@blaze_tools//tools/cpp/link_dynamic_library");
        }
      };

  private BazelMockCcSupport() {}

  private static final ImmutableList<String> CROSSTOOL_ARCHS =
      ImmutableList.of("piii", "k8", "armeabi-v7a", "ppc");

  protected static void createBasePackage(MockToolsConfig config) throws IOException {
    config.create(
        "base/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "cc_library(name = 'system_malloc', linkstatic = 1)",
        "cc_library(name = 'base', srcs=['timestamp.h'])");
    if (config.isRealFileSystem()) {
      config.linkTool("base/timestamp.h");
    } else {
      config.create("base/timestamp.h", "");
    }
  }

  @Override
  protected String getRealFilesystemCrosstoolTopPath() {
    assert false;
    return null;
  }

  @Override
  protected String[] getRealFilesystemTools(String crosstoolTop) {
    assert false;
    return null;
  }

  @Override
  protected ImmutableList<String> getCrosstoolArchs() {
    return CROSSTOOL_ARCHS;
  }

  @Override
  public void setup(MockToolsConfig config) throws IOException {
    config.create(
        "/bazel_tools_workspace/tools/cpp/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "toolchain_type(name = 'toolchain_type')",
        "cc_library(name = 'malloc')",
        "cc_toolchain_suite(",
        "    name = 'toolchain',",
        "    toolchains = {",
        "      'local': ':cc-compiler-local',",
        "      'k8': ':cc-compiler-k8',",
        "      'piii': ':cc-compiler-piii-gcc-4.4.0',",
        "      'darwin': ':cc-compiler-darwin',",
        "      'ios_x86_64': ':cc-compiler-ios_x86_64',",
        "      'armeabi-v7a': ':cc-compiler-armeabi-v7a',",
        "      'x64_windows': ':cc-compiler-x64_windows',",
        "      'ppc': ':cc-compiler-ppc',",
        "      'local|compiler': ':cc-compiler-local',",
        "      'k8|compiler': ':cc-compiler-k8',",
        "      'k8|compiler_no_dyn_linker': ':cc-no-dyn-linker-k8',",
        "      'piii|compiler': ':cc-compiler-piii-gcc-4.4.0',",
        "      'darwin|compiler': ':cc-compiler-darwin',",
        "      'darwin|compiler_no_dyn_linker': ':cc-no-dyn-linker-darwin',",
        "      'ios_x86_64|compiler': ':cc-compiler-ios_x86_64',",
        "      'armeabi-v7a|compiler': ':cc-compiler-armeabi-v7a',",
        "      'x64_windows|windows_msys64': ':cc-compiler-x64_windows',",
        "      'x64_windows|compiler_no_dyn_linker': ':cc-no-dyn-linker-x64_windows',",
        "      'ppc|compiler': ':cc-compiler-ppc',",
        "    })",
        "cc_toolchain(name = 'cc-compiler-local', all_files = ':empty', compiler_files = ':empty',",
        "    toolchain_identifier = 'toolchain-identifier-local',",
        "    cpu = 'local',",
        "    compiler = 'compiler',",
        "    dwp_files = ':empty',",
        "    dynamic_runtime_lib = ':empty', ",
        "    ar_files = ':empty', as_files = ':empty', linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_lib = ':empty', strip_files = ':empty',",
        ")",
        "toolchain(name = 'cc-toolchain-local',",
        // Needs to be compatible with all execution environments for tests to work properly.
        "    exec_compatible_with = [],",
        "    target_compatible_with = [",
        "        '@bazel_tools//platforms:x86_64',",
        "        '@bazel_tools//platforms:linux',",
        "    ],",
        "    toolchain = ':cc-compiler-local',",
        "    toolchain_type = ':toolchain_type',",
        ")",
        "cc_toolchain(name = 'cc-compiler-k8', all_files = ':empty', compiler_files = ':empty',",
        "    toolchain_identifier = 'toolchain-identifier-k8',",
        "    cpu = 'k8',",
        "    compiler = 'compiler',",
        "    dwp_files = ':empty',",
        "    dynamic_runtime_lib = ':empty', ",
        "    ar_files = ':empty', as_files = ':empty', linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_lib = ':empty', strip_files = ':empty',",
        ")",
        "toolchain(name = 'cc-toolchain-k8',",
        // Needs to be compatible with all execution environments for tests to work properly.
        "    exec_compatible_with = [],",
        "    target_compatible_with = [",
        "        '@bazel_tools//platforms:x86_64',",
        "        '@bazel_tools//platforms:linux',",
        "    ],",
        "    toolchain = ':cc-compiler-k8',",
        "    toolchain_type = ':toolchain_type',",
        ")",
        "cc_toolchain(name = 'cc-no-dyn-linker-k8', all_files = ':empty', ",
        "    compiler_files = ':empty', cpu = 'k8', compiler = 'compiler_no_dyn_linker', ",
        "    dwp_files = ':empty', dynamic_runtime_lib = ':empty', ",
        "    ar_files = ':empty', as_files = ':empty', linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_lib = ':empty', strip_files = ':empty',",
        ")",
        "toolchain(name = 'cc-toolchain-no-dyn-linker-k8',",
        // Needs to be compatible with all execution environments for tests to work properly.
        "    exec_compatible_with = [],",
        "    target_compatible_with = [",
        "        '@bazel_tools//platforms:x86_64',",
        "        '@bazel_tools//platforms:linux',",
        "    ],",
        "    toolchain = ':cc-no-dyn-linker-k8',",
        "    toolchain_type = ':toolchain_type',",
        ")",
        "cc_toolchain(name = 'cc-compiler-ppc', all_files = ':empty', compiler_files = ':empty',",
        "    cpu = 'ppc',",
        "    compiler = 'compiler',",
        "    dwp_files = ':empty',",
        "    dynamic_runtime_lib = ':empty', ",
        "    ar_files = ':empty', as_files = ':empty', linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_lib = ':empty', strip_files = ':empty',",
        ")",
        "toolchain(name = 'cc-toolchain-ppc',",
        // Needs to be compatible with all execution environments for tests to work properly.
        "    exec_compatible_with = [],",
        "    target_compatible_with = [",
        "        '@bazel_tools//platforms:ppc',",
        "        '@bazel_tools//platforms:linux',",
        "    ],",
        "    toolchain = ':cc-compiler-ppc',",
        "    toolchain_type = ':toolchain_type',",
        ")",
        "cc_toolchain(name = 'cc-compiler-piii-gcc-4.4.0',",
        "    all_files = ':all-files-piii',",
        "    compiler_files = ':compiler-files-piii',",
        "    cpu = 'piii',",
        "    compiler = 'compiler',",
        "    dwp_files = ':empty',",
        "    dynamic_runtime_lib = ':empty', ",
        "    ar_files = ':empty', as_files = ':empty', linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_lib = ':empty', strip_files = ':empty',",
        ")",
        "toolchain(name = 'cc-toolchain-piii',",
        // Needs to be compatible with all execution environments for tests to work properly.
        "    exec_compatible_with = [],",
        "    target_compatible_with = [",
        "        '@bazel_tools//platforms:x86_32',",
        "        '@bazel_tools//platforms:linux',",
        "    ],",
        "    toolchain = ':cc-compiler-piii-gcc-4.4.0',",
        "    toolchain_type = ':toolchain_type',",
        ")",
        "cc_toolchain(name = 'cc-compiler-darwin', all_files = ':empty', ",
        "    compiler_files = ':empty',",
        "    cpu = 'darwin',",
        "    compiler = 'compiler',",
        "    dwp_files = ':empty',",
        "    dynamic_runtime_lib = ':empty', ",
        "    ar_files = ':empty', as_files = ':empty', linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_lib = ':empty', strip_files = ':empty',",
        ")",
        "toolchain(name = 'cc-toolchain-darwin',",
        // Needs to be compatible with all execution environments for tests to work properly.
        "    exec_compatible_with = [],",
        "    target_compatible_with = [",
        "        '@bazel_tools//platforms:x86_64',",
        "        '@bazel_tools//platforms:osx',",
        "    ],",
        "    toolchain = ':cc-compiler-darwin',",
        "    toolchain_type = ':toolchain_type',",
        ")",
        "cc_toolchain(name = 'cc-no-dyn-linker-darwin', all_files = ':empty', ",
        "    compiler_files = ':empty',",
        "    cpu = 'darwin',",
        "    compiler = 'compiler_no_dyn_linker',",
        "    dwp_files = ':empty', dynamic_runtime_lib = ':empty', ",
        "    ar_files = ':empty', as_files = ':empty', linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_lib = ':empty', strip_files = ':empty',",
        ")",
        "toolchain(name = 'cc-toolchain-no-dyn-linker-darwin',",
        // Needs to be compatible with all execution environments for tests to work properly.
        "    exec_compatible_with = [],",
        "    target_compatible_with = [",
        "        '@bazel_tools//platforms:x86_64',",
        "        '@bazel_tools//platforms:osx',",
        "    ],",
        "    toolchain = ':cc-no-dyn-linker-darwin',",
        "    toolchain_type = ':toolchain_type',",
        ")",
        "cc_toolchain(name = 'cc-compiler-ios_x86_64', all_files = ':empty', ",
        "    compiler_files = ':empty',",
        "    cpu = 'ios_x86_64',",
        "    compiler = 'compiler',",
        "    dwp_files = ':empty',",
        "    dynamic_runtime_lib = ':empty', ",
        "    ar_files = ':empty', as_files = ':empty', linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_lib = ':empty', strip_files = ':empty',",
        ")",
        "toolchain(name = 'cc-toolchain-ios_x86_64',",
        // Needs to be compatible with all execution environments for tests to work properly.
        "    exec_compatible_with = [],",
        "    target_compatible_with = [",
        "        '@bazel_tools//platforms:x86_64',",
        "        '@bazel_tools//platforms:ios',",
        "    ],",
        "    toolchain = ':cc-compiler-ios_x86_64',",
        "    toolchain_type = ':toolchain_type',",
        ")",
        "cc_toolchain(name = 'cc-compiler-armeabi-v7a', all_files = ':empty', ",
        "    compiler_files = ':empty',",
        "    cpu = 'armeabi-v7a',",
        "    compiler = 'compiler',",
        "    dwp_files = ':empty',",
        "    dynamic_runtime_lib = ':empty', ",
        "    ar_files = ':empty', as_files = ':empty', linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_lib = ':empty', strip_files = ':empty',",
        ")",
        "toolchain(name = 'cc-toolchain-armeabi-v7a',",
        // Needs to be compatible with all execution environments for tests to work properly.
        "    exec_compatible_with = [],",
        "    target_compatible_with = [",
        "        '@bazel_tools//platforms:arm',",
        "        '@bazel_tools//platforms:android',",
        "    ],",
        "    toolchain = ':cc-compiler-armeabi-v7a',",
        "    toolchain_type = ':toolchain_type',",
        ")",
        "cc_toolchain(name = 'cc-compiler-x64_windows', all_files = ':empty', ",
        "    compiler_files = ':empty',",
        "    cpu = 'x64_windows',",
        "    compiler = 'windows_msys64',",
        "    dwp_files = ':empty',",
        "    dynamic_runtime_lib = ':empty', ",
        "    ar_files = ':empty', as_files = ':empty', linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_lib = ':empty', strip_files = ':empty',",
        ")",
        "toolchain(name = 'cc-toolchain-x64_windows',",
        // Needs to be compatible with all execution environments for tests to work properly.
        "    exec_compatible_with = [],",
        "    target_compatible_with = [",
        "        '@bazel_tools//platforms:x86_64',",
        "        '@bazel_tools//platforms:windows',",
        "    ],",
        "    toolchain = ':cc-compiler-x64_windows',",
        "    toolchain_type = ':toolchain_type',",
        ")",
        "cc_toolchain(name = 'cc-no-dyn-linker-x64_windows', all_files = ':empty', ",
        "    compiler_files = ':empty',",
        "    cpu = 'x64_windows',",
        "    compiler = 'compiler_no_dyn_linker',",
        "    dwp_files = ':empty',",
        "    dynamic_runtime_lib = ':empty', ",
        "    ar_files = ':empty', as_files = ':empty', linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_lib = ':empty', strip_files = ':empty',",
        ")",
        "toolchain(name = 'cc-toolchain-no-dyn-linker-x64_windows',",
        // Needs to be compatible with all execution environments for tests to work properly.
        "    exec_compatible_with = [],",
        "    target_compatible_with = [",
        "        '@bazel_tools//platforms:x86_64',",
        "        '@bazel_tools//platforms:windows',",
        "    ],",
        "    toolchain = ':cc-no-dyn-linker-x64_windows',",
        "    toolchain_type = ':toolchain_type',",
        ")",
        "filegroup(",
        "    name = 'interface_library_builder',",
        "    srcs = ['build_interface_so'],",
        ")",
        "filegroup(",
        "    name = 'link_dynamic_library',",
        "    srcs = ['link_dynamic_library.sh'],",
        ")",
        "cc_toolchain_alias(name = 'current_cc_toolchain')",
        "filegroup(",
        "    name = 'crosstool',",
        "    srcs = [':current_cc_toolchain'],",
        ")");
    config.create(
        "/bazel_tools_workspace/tools/cpp/CROSSTOOL",
        readCrosstoolFile());
    if (config.isRealFileSystem()) {
      config.linkTool("tools/cpp/link_dynamic_library.sh");
    } else {
      config.create("tools/cpp/link_dynamic_library.sh", "");
    }
    MockPlatformSupport.setup(
        config, "/bazel_tools_workspace/platforms", "/local_config_platform_workspace");
  }

  @Override
  public String getMockCrosstoolVersion() {
    return "gcc-4.4.0-glibc-2.3.6";
  }

  @Override
  public Label getMockCrosstoolLabel() {
    return Label.parseAbsoluteUnchecked("@bazel_tools//tools/cpp:toolchain");
  }

  @Override
  public String readCrosstoolFile() throws IOException {
    return ResourceLoader.readFromResources(MOCK_CROSSTOOL_PATH);
  }

  @Override
  public String getMockCrosstoolPath() {
    return "/bazel_tools_workspace/tools/cpp/";
  }

  @Override
  public Predicate<String> labelNameFilter() {
    return CC_LABEL_NAME_FILTER;
  }
}

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
          return !label.startsWith("@blaze_tools//tools/cpp/stl")
              && !label.startsWith("@blaze_tools//tools/cpp/link_dynamic_library");
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
        "cc_library(name = 'stl')",
        "toolchain_type(name = 'toolchain_type')",
        "cc_library(name = 'malloc')",
        "cc_toolchain_suite(",
        "    name = 'toolchain',",
        "    toolchains = {",
        "      'local|compiler': ':cc-compiler-local',",
        "      'k8|compiler': ':cc-compiler-k8',",
        "      'piii|compiler': ':cc-compiler-piii',",
        "      'darwin|compiler': ':cc-compiler-darwin',",
        "      'ios_x86_64|compiler': ':cc-compiler-ios_x86_64',",
        "      'armeabi-v7a|compiler': ':cc-compiler-armeabi-v7a',",
        "      'x64_windows|windows_msys64': ':cc-compiler-x64_windows',",
        "      'ppc|compiler': ':cc-compiler-ppc',",
        "    })",
        "cc_toolchain(name = 'cc-compiler-k8', all_files = ':empty', compiler_files = ':empty',",
        "    cpu = 'local', dwp_files = ':empty', dynamic_runtime_libs = [':empty'], ",
        "    linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_libs = [':empty'], strip_files = ':empty',",
        ")",
        "cc_toolchain(name = 'cc-compiler-ppc', all_files = ':empty', compiler_files = ':empty',",
        "    cpu = 'ppc', dwp_files = ':empty', dynamic_runtime_libs = [':empty'], ",
        "    linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_libs = [':empty'], strip_files = ':empty',",
        ")",
        "cc_toolchain(name = 'cc-compiler-piii', all_files = ':all-files-piii',",
        "    compiler_files = ':compiler-files-piii',",
        "    cpu = 'local', dwp_files = ':empty', dynamic_runtime_libs = [':empty'], ",
        "    linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_libs = [':empty'], strip_files = ':empty',",
        ")",
        "cc_toolchain(name = 'cc-compiler-darwin', all_files = ':empty', ",
        "    compiler_files = ':empty',",
        "    cpu = 'local', dwp_files = ':empty', dynamic_runtime_libs = [':empty'], ",
        "    linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_libs = [':empty'], strip_files = ':empty',",
        ")",
        "cc_toolchain(name = 'cc-compiler-ios_x86_64', all_files = ':empty', ",
        "    compiler_files = ':empty',",
        "    cpu = 'local', dwp_files = ':empty', dynamic_runtime_libs = [':empty'], ",
        "    linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_libs = [':empty'], strip_files = ':empty',",
        ")",
        "cc_toolchain(name = 'cc-compiler-armeabi-v7a', all_files = ':empty', ",
        "    compiler_files = ':empty',",
        "    cpu = 'local', dwp_files = ':empty', dynamic_runtime_libs = [':empty'], ",
        "    linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_libs = [':empty'], strip_files = ':empty',",
        ")",
        "cc_toolchain(name = 'cc-compiler-x64_windows', all_files = ':empty', ",
        "    compiler_files = ':empty',",
        "    cpu = 'local', dwp_files = ':empty', dynamic_runtime_libs = [':empty'], ",
        "    linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_libs = [':empty'], strip_files = ':empty',",
        ")",
        "filegroup(",
        "    name = 'interface_library_builder',",
        "    srcs = ['build_interface_so'],",
        ")",
        "filegroup(",
        "    name = 'link_dynamic_library',",
        "    srcs = ['link_dynamic_library.sh'],",
        ")",
        "filegroup(name = 'toolchain_category')",
        "toolchain(",
        "   name = 'toolchain_cc-compiler-piii',",
        "   toolchain_type = ':toolchain_category',",
        "   toolchain = '//third_party/crosstool/mock:cc-compiler-piii',",
        "   target_compatible_with = [':mock_value'],",
        ")",
        "toolchain(",
        "   name = 'dummy_cc_toolchain',",
        "   toolchain_type = ':toolchain_category',",
        "   toolchain = ':dummy_cc_toolchain_impl',",
        ")",
        "load(':dummy_toolchain.bzl', 'dummy_toolchain')",
        "dummy_toolchain(name = 'dummy_cc_toolchain_impl')");
    config.create(
        "/bazel_tools_workspace/tools/cpp/dummy_toolchain.bzl",
        "def _dummy_toolchain_impl(ctx):",
        "   toolchain = platform_common.ToolchainInfo()",
        "   return [toolchain]",
        "dummy_toolchain = rule(_dummy_toolchain_impl, attrs = {})");
    config.create(
        "/bazel_tools_workspace/tools/cpp/CROSSTOOL",
        readCrosstoolFile());
    if (config.isRealFileSystem()) {
      config.linkTool("tools/cpp/link_dynamic_library.sh");
    } else {
      config.create("tools/cpp/link_dynamic_library.sh", "");
    }
    MockObjcSupport.setup(config);
    MockPlatformSupport.setup(config, "/bazel_tools_workspace/platforms");
    MockPlatformSupport.setup(config, "/bazel_tools_workspace/");
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
  public String getMockLibcPath() {
    return "tools/cpp/libc";
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

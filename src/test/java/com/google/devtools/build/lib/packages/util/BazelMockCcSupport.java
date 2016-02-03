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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.io.ByteStreams;

import java.io.IOException;
import java.io.InputStream;

/**
 * Bazel implementation of {@link MockCcSupport}
 */
public final class BazelMockCcSupport extends MockCcSupport {

  public static final BazelMockCcSupport INSTANCE = new BazelMockCcSupport();
  /** Filter to remove implicit dependencies of C/C++ rules. */
  private static final Predicate<String> CC_LABEL_NAME_FILTER =
      new Predicate<String>() {
        @Override
        public boolean apply(String label) {
          return !label.startsWith("@blaze_tools//tools/cpp/stl");
        }
      };

  private BazelMockCcSupport() {}

  private static final ImmutableList<String> CROSSTOOL_ARCHS =
      ImmutableList.of("piii", "k8", "armeabi-v7a");

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
        "cc_library(name = 'stl')",
        "cc_library(name = 'malloc')",
        "filegroup(name = 'toolchain', ",
        "    srcs = [':cc-compiler-local', ':cc-compiler-darwin', ':cc-compiler-piii',",
        "            ':cc-compiler-armeabi-v7a', ':empty'],",
        ")",
        "cc_toolchain(name = 'cc-compiler-k8', all_files = ':empty', compiler_files = ':empty',",
        "    cpu = 'local', dwp_files = ':empty', dynamic_runtime_libs = [':empty'], ",
        "    linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_libs = [':empty'], strip_files = ':empty',",
        ")",
        "cc_toolchain(name = 'cc-compiler-piii', all_files = ':empty', compiler_files = ':empty',",
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
        "cc_toolchain(name = 'cc-compiler-armeabi-v7a', all_files = ':empty', ",
        "    compiler_files = ':empty',",
        "    cpu = 'local', dwp_files = ':empty', dynamic_runtime_libs = [':empty'], ",
        "    linker_files = ':empty',",
        "    module_map = 'crosstool.cppmap', supports_header_parsing = 1,",
        "    objcopy_files = ':empty', static_runtime_libs = [':empty'], strip_files = ':empty',",
        ")");

    config.create(
        "/bazel_tools_workspace/tools/cpp/CROSSTOOL",
        readFromResources("com/google/devtools/build/lib/MOCK_CROSSTOOL"));
    config.create(
        "/bazel_tools_workspace/tools/objc/BUILD",
        "xcode_config(name = 'host_xcodes')");
  }

  @Override
  protected String getMockCrosstoolVersion() {
    return "gcc-4.4.0-glibc-2.3.6";
  }

  @Override
  protected String readCrosstoolFile() throws IOException {
    return readFromResources("com/google/devtools/build/lib/MOCK_CROSSTOOL");
  }

  public static String readFromResources(String filename) throws IOException {
    InputStream in = BazelMockCcSupport.class.getClassLoader().getResourceAsStream(filename);
    return new String(ByteStreams.toByteArray(in), UTF_8);
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

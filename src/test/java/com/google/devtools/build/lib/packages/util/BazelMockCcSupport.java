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

  /** Filter to remove implicit dependencies of C/C++ rules. */
  private static final Predicate<String> CC_LABEL_NAME_FILTER =
      new Predicate<String>() {
        @Override
        public boolean apply(String label) {
          return !label.startsWith("//tools/cpp");
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
    writeMacroFile(config);
    setupRulesCc(config);
    setupCcToolchainConfig(config);
  }

  @Override
  public Label getMockCrosstoolLabel() {
    return Label.parseAbsoluteUnchecked("@bazel_tools//tools/cpp:toolchain");
  }

  @Override
  public String getMockCrosstoolPath() {
    return "bazel_tools_workspace/tools/cpp/";
  }

  @Override
  public Predicate<String> labelNameFilter() {
    return CC_LABEL_NAME_FILTER;
  }
}

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
import com.google.devtools.build.lib.testutil.TestConstants;
import java.io.IOException;

/**
 * Bazel implementation of {@link MockCcSupport}
 */
public final class BazelMockCcSupport extends MockCcSupport {
  public static final BazelMockCcSupport INSTANCE = new BazelMockCcSupport();

  /** Filter to remove implicit dependencies of C/C++ rules. */
  private static final boolean isNotCcLabel(String label) {
    return !label.startsWith("//tools/cpp");
  }

  private BazelMockCcSupport() {}

  private static final ImmutableList<String> CROSSTOOL_ARCHS =
      ImmutableList.of("piii", "k8", "armeabi-v7a", "ppc");

  @Override
  protected String getRealFilesystemCrosstoolTopPath() {
    // TODO(b/195425240): Make real-filesystem mode work.
    return "";
  }

  @Override
  protected String[] getRealFilesystemTools(String crosstoolTop) {
    // TODO(b/195425240): Make real-filesystem mode work.
    return new String[0];
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
    createDummyCppPackages(config);
    createParseHeadersAndLayeringCheckWhitelist(config);
    createStarlarkLooseHeadersWhitelist(config, "//...");
  }

  @Override
  public Label getMockCrosstoolLabel() {
    return Label.parseAbsoluteUnchecked("@bazel_tools//tools/cpp:toolchain");
  }

  @Override
  public String getMockCrosstoolPath() {
    return "embedded_tools/tools/cpp/";
  }

  @Override
  public Predicate<String> labelNameFilter() {
    return BazelMockCcSupport::isNotCcLabel;
  }

  /** Creates bare-minimum filesystem state to support cpp rules. */
  private static void createDummyCppPackages(MockToolsConfig config) throws IOException {
    if (config.isRealFileSystem()) {
      // TODO(b/195425240): Make real-filesystem test mode work in bazel - for now we fake out the
      //  bare minimum targets to get by in at least the loading phase.
      config.append(
          TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/cpp/BUILD",
          "exports_files(['toolchain', 'grep-includes', 'malloc'])");
      config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/cpp/toolchain", "");
      config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/cpp/grep-includes", "");
      config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/cpp/malloc", "");
    }
  }
}

// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Joiner;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for cc_binary with split functions. */
@RunWith(JUnit4.class)
public class CcBinarySplitFunctionsTest extends BuildViewTestCase {

  @Before
  public void createBasePkg() throws IOException {
    scratch.overwriteFile(
        "base/BUILD", "cc_library(name = 'system_malloc', visibility = ['//visibility:public'])");
  }

  private Action getPredecessorByInputName(Action action, String str) {
    for (Artifact a : action.getInputs().toList()) {
      if (a.getExecPathString().contains(str)) {
        return getGeneratingAction(a);
      }
    }
    return null;
  }

  private LtoBackendAction setupAndRunToolchainActions(String... config) throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(
                    CppRuleClasses.THIN_LTO,
                    MockCcSupport.HOST_AND_NONHOST_CONFIGURATION_FEATURES,
                    CppRuleClasses.FDO_OPTIMIZE,
                    CppRuleClasses.SUPPORTS_START_END_LIB,
                    CppRuleClasses.ENABLE_FDO_SPLIT_FUNCTIONS,
                    MockCcSupport.FDO_SPLIT_FUNCTIONS,
                    MockCcSupport.SPLIT_FUNCTIONS));

    List<String> testConfig =
        Lists.newArrayList("--fdo_optimize=/pkg/profile.zip", "--compilation_mode=opt");
    Collections.addAll(testConfig, config);
    useConfiguration(Iterables.toArray(testConfig, String.class));

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto/" + rootExecPath + "/pkg/_objs/bin/binfile.o");

    // We should have a ThinLTO backend action.
    assertThat(backendAction).isNotNull();

    return backendAction;
  }

  /**
   * Tests that split_functions is enabled for FDO with LLVM with --features=fdo_split_functions.
   */
  @Test
  public void fdoImplicitSplitFunctions() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "package(features = ['thin_lto'])",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          malloc = '//base:system_malloc')");
    scratch.file("pkg/binfile.cc", "int main() {}");
    scratch.file("pkg/profile.zip", "");

    LtoBackendAction backendAction = setupAndRunToolchainActions("--features=fdo_split_functions");

    assertThat(Joiner.on(" ").join(backendAction.getArguments()))
        .containsMatch("-fsplit-machine-functions");
    assertThat(Joiner.on(" ").join(backendAction.getArguments()))
        .containsMatch("-DBUILD_PROPELLER_TYPE=\"split\"");
  }

  /**
   * Tests that split_functions is not enabled for FDO with LLVM without
   * --features=fdo_split_functions.
   */
  @Test
  public void fdoNoImplicitSplitFunctions() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "package(features = ['thin_lto'])",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          malloc = '//base:system_malloc')");
    scratch.file("pkg/binfile.cc", "int main() {}");
    scratch.file("pkg/profile.zip", "");

    LtoBackendAction backendAction = setupAndRunToolchainActions();

    assertThat(Joiner.on(" ").join(backendAction.getArguments()))
        .doesNotContain("-fsplit-machine-functions");
  }

  /**
   * Tests that split_functions is not enabled for FDO with LLVM when --features=fdo_split_functions
   * is overridden by --features=-split_functions.
   */
  @Test
  public void fdoImplicitSplitFunctionsDisabledOption() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "package(features = ['thin_lto'])",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          malloc = '//base:system_malloc')");
    scratch.file("pkg/binfile.cc", "int main() {}");
    scratch.file("pkg/profile.zip", "");

    LtoBackendAction backendAction =
        setupAndRunToolchainActions(
            "--features=fdo_split_functions", "--features=-split_functions");

    assertThat(Joiner.on(" ").join(backendAction.getArguments()))
        .doesNotContain("-fsplit-machine-functions");
  }

  /**
   * Tests that split_functions is not enabled for FDO with LLVM when --features=fdo_split_functions
   * is overridden by --features=-split_functions in the build rule.
   */
  @Test
  public void fdoImplicitSplitFunctionsDisabledBuild() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "package(features = ['thin_lto'])",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          features = ['-split_functions'],",
        "          malloc = '//base:system_malloc')");
    scratch.file("pkg/binfile.cc", "int main() {}");
    scratch.file("pkg/profile.zip", "");

    LtoBackendAction backendAction = setupAndRunToolchainActions("--features=fdo_split_functions");

    assertThat(Joiner.on(" ").join(backendAction.getArguments()))
        .doesNotContain("-fsplit-machine-functions");
  }

  /** Tests that using propeller_optimize automatically disables implicit split functions. */
  @Test
  public void propellerOptimizeDisablesImplicitSplitFunctions() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "package(features = ['thin_lto'])",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          malloc = '//base:system_malloc')");
    scratch.file("pkg/binfile.cc", "int main() {}");
    scratch.file("pkg/profile.zip", "");

    LtoBackendAction backendAction =
        setupAndRunToolchainActions(
            "--features=fdo_split_functions",
            "--propeller_optimize_absolute_cc_profile=/tmp/cc.txt");

    assertThat(Joiner.on(" ").join(backendAction.getArguments()))
        .containsMatch("-fbasic-block-sections=list=");
    assertThat(Joiner.on(" ").join(backendAction.getArguments()))
        .containsMatch("-DBUILD_PROPELLER_TYPE=\"full\"");
    assertThat(Joiner.on(" ").join(backendAction.getArguments()))
        .doesNotMatch("-DBUILD_PROPELLER_TYPE=\"split\"");
    assertThat(Joiner.on(" ").join(backendAction.getArguments()))
        .doesNotMatch("-fsplit-machine-functions");
  }

  /**
   * Tests that split_functions is not enabled for FDO with LLVM when --features=fdo_split_functions
   * is overridden by --features=-split_functions in the package.
   */
  @Test
  public void fdoImplicitSplitFunctionsDisabledPackage() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "package(features = ['thin_lto', '-split_functions'])",
        "",
        "cc_binary(name = 'bin',",
        "          srcs = ['binfile.cc', ],",
        "          malloc = '//base:system_malloc')");
    scratch.file("pkg/binfile.cc", "int main() {}");
    scratch.file("pkg/profile.zip", "");

    LtoBackendAction backendAction = setupAndRunToolchainActions("--features=fdo_split_functions");

    assertThat(Joiner.on(" ").join(backendAction.getArguments()))
        .doesNotContain("-fsplit-machine-functions");
  }
}

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
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
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

/** Tests for cc_binary with fsafdo features. */
@RunWith(JUnit4.class)
public class CcBinaryFSAFDOTest extends BuildViewTestCase {

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
                    CppRuleClasses.AUTOFDO,
                    CppRuleClasses.ENABLE_FSAFDO,
                    MockCcSupport.HOST_AND_NONHOST_CONFIGURATION_FEATURES,
                    CppRuleClasses.SUPPORTS_START_END_LIB,
                    MockCcSupport.IMPLICIT_FSAFDO,
                    MockCcSupport.FSAFDO));

    List<String> testConfig =
        Lists.newArrayList("--fdo_optimize=/pkg/profile.afdo", "--compilation_mode=opt");
    Collections.addAll(testConfig, config);
    useConfiguration(Iterables.toArray(testConfig, String.class));

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();

    SpawnAction linkAction = (SpawnAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto/" + rootExecPath + "/pkg/_objs/bin/binfile.o");

    // We should have a ThinLTO backend action.
    assertThat(backendAction).isNotNull();

    return backendAction;
  }

  /** Tests that fsafdo is enabled with LLVM with --features=implicit_fsafdo. */
  @Test
  public void fsafdoEnabledWithImplicit() throws Exception {
    scratch.file(
        "pkg/BUILD",
        """
        package(features = ["thin_lto"])

        cc_binary(
            name = "bin",
            srcs = ["binfile.cc"],
            malloc = "//base:system_malloc",
        )
        """);
    scratch.file("pkg/binfile.cc", "int main() {}");

    LtoBackendAction backendAction = setupAndRunToolchainActions("--features=implicit_fsafdo");

    assertThat(Joiner.on(" ").join(backendAction.getArguments())).containsMatch("-fsafdo");
  }

  /** Tests that fsafdo is enabled with LLVM with --features=-implicit_fsafdo --features=fsafdo. */
  @Test
  public void fsafdoEnabledWithFeatureWithoutImplicit() throws Exception {
    scratch.file(
        "pkg/BUILD",
        """
        package(features = ["thin_lto"])

        cc_binary(
            name = "bin",
            srcs = ["binfile.cc"],
            malloc = "//base:system_malloc",
        )
        """);
    scratch.file("pkg/binfile.cc", "int main() {}");

    LtoBackendAction backendAction =
        setupAndRunToolchainActions("--features=-implicit_fsafdo", "--features=fsafdo");

    assertThat(Joiner.on(" ").join(backendAction.getArguments())).containsMatch("-fsafdo");
  }

  /** Tests that fsafdo is enabled with LLVM with --features=fsafdo. */
  @Test
  public void fsafdoEnabledWithExplicitFeature() throws Exception {
    scratch.file(
        "pkg/BUILD",
        """
        package(features = ["thin_lto"])

        cc_binary(
            name = "bin",
            srcs = ["binfile.cc"],
            malloc = "//base:system_malloc",
        )
        """);
    scratch.file("pkg/binfile.cc", "int main() {}");

    LtoBackendAction backendAction = setupAndRunToolchainActions("--features=fsafdo");

    assertThat(Joiner.on(" ").join(backendAction.getArguments())).containsMatch("-fsafdo");
  }

  /** Tests that FSAFDO is not enabled in LLVM without --features=implicit_fsafdo. */
  @Test
  public void fsafdoDisabledWithFeatureWithoutImplicit() throws Exception {
    scratch.file(
        "pkg/BUILD",
        """
        package(features = ["thin_lto"])

        cc_binary(
            name = "bin",
            srcs = ["binfile.cc"],
            malloc = "//base:system_malloc",
        )
        """);
    scratch.file("pkg/binfile.cc", "int main() {}");

    LtoBackendAction backendAction = setupAndRunToolchainActions();

    assertThat(Joiner.on(" ").join(backendAction.getArguments())).doesNotContain("-fsafdo");
  }

  /**
   * Tests that fsafdo is not enabled in LLVM with --features=implicit_fsafdo and
   * --features=-fsafdo.
   */
  @Test
  public void fsafdoDisabledWithExplicitFeature() throws Exception {
    scratch.file(
        "pkg/BUILD",
        """
        package(features = ["thin_lto"])

        cc_binary(
            name = "bin",
            srcs = ["binfile.cc"],
            malloc = "//base:system_malloc",
        )
        """);
    scratch.file("pkg/binfile.cc", "int main() {}");

    LtoBackendAction backendAction =
        setupAndRunToolchainActions("--features=implicit_fsafdo", "--features=-fsafdo");

    assertThat(Joiner.on(" ").join(backendAction.getArguments())).doesNotContain("-fsafdo");
  }

  /** Test that fsafdo is not enable with --features=fsafdo without autofdo. */
  @Test
  public void fsafdoDisabledForNonAutoFDO() throws Exception {
    scratch.file(
        "pkg/BUILD",
        """
        package(features = ["thin_lto"])

        cc_binary(
            name = "bin",
            srcs = ["binfile.cc"],
            malloc = "//base:system_malloc",
        )
        """);
    scratch.file("pkg/binfile.cc", "int main() {}");

    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(
                    CppRuleClasses.THIN_LTO,
                    CppRuleClasses.ENABLE_FSAFDO,
                    MockCcSupport.HOST_AND_NONHOST_CONFIGURATION_FEATURES,
                    CppRuleClasses.SUPPORTS_START_END_LIB,
                    MockCcSupport.IMPLICIT_FSAFDO,
                    MockCcSupport.FSAFDO));

    List<String> testConfig = Lists.newArrayList("--compilation_mode=opt");
    Collections.addAll(testConfig, "--features=fsafdo");
    useConfiguration(Iterables.toArray(testConfig, String.class));

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();
    SpawnAction linkAction = (SpawnAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto/" + rootExecPath + "/pkg/_objs/bin/binfile.o");

    // We should have a ThinLTO backend action.
    assertThat(backendAction).isNotNull();

    assertThat(Joiner.on(" ").join(backendAction.getArguments())).doesNotContain("-fsafdo");
  }

  /** Test that fsafdo is not enable with --features=fsafdo for XBinaryFDO. */
  @Test
  public void fsafdoDisabledForXFdo() throws Exception {
    scratch.file(
        "pkg/BUILD",
        """
        package(features = ["thin_lto"])

        cc_binary(
            name = "bin",
            srcs = ["binfile.cc"],
            malloc = "//base:system_malloc",
        )

        fdo_profile(
            name = "out.xfdo",
            profile = "profiles.xfdo",
        )
        """);
    scratch.file("pkg/binfile.cc", "int main() {}");

    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(
                    CppRuleClasses.THIN_LTO,
                    CppRuleClasses.XBINARYFDO,
                    CppRuleClasses.ENABLE_XFDO_THINLTO,
                    CppRuleClasses.ENABLE_FSAFDO,
                    MockCcSupport.HOST_AND_NONHOST_CONFIGURATION_FEATURES,
                    CppRuleClasses.SUPPORTS_START_END_LIB,
                    MockCcSupport.IMPLICIT_FSAFDO,
                    MockCcSupport.FSAFDO,
                    MockCcSupport.XFDO_IMPLICIT_THINLTO));

    List<String> testConfig =
        Lists.newArrayList("--xbinary_fdo=//pkg:out.xfdo", "--compilation_mode=opt");
    Collections.addAll(testConfig, "--features=fsafdo");
    useConfiguration(Iterables.toArray(testConfig, String.class));

    Artifact binArtifact = getFilesToBuild(getConfiguredTarget("//pkg:bin")).getSingleton();
    String rootExecPath = binArtifact.getRoot().getExecPathString();
    SpawnAction linkAction = (SpawnAction) getGeneratingAction(binArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(binArtifact);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "pkg/bin.lto/" + rootExecPath + "/pkg/_objs/bin/binfile.o");

    // We should have a ThinLTO backend action.
    assertThat(backendAction).isNotNull();

    assertThat(Joiner.on(" ").join(backendAction.getArguments())).doesNotContain("-fsafdo");
  }
}

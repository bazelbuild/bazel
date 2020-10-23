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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.testing.NullPointerTester;
import com.google.common.testing.NullPointerTester.Visibility;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.cpp.CcCompilationHelper.SourceCategory;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CcCompilationHelper}. */
@RunWith(JUnit4.class)
public final class CcCompilationHelperTest extends BuildViewTestCase {

  @Test
  public void testConstructorThrowsNPE() throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget("a", "b", "cc_library(name = 'b', srcs = [],)");
    RuleContext ruleContext = getRuleContext(target);
    CcToolchainProvider ccToolchain =
        CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext);
    FdoContext fdoContext = ccToolchain.getFdoContext();
    Artifact grepIncludes = getBinArtifact("grep_includes", target);
    NullPointerTester tester =
        new NullPointerTester()
            .setDefault(RuleContext.class, ruleContext)
            .setDefault(CcCommon.class, new CcCommon(ruleContext))
            .setDefault(CppSemantics.class, MockCppSemantics.INSTANCE)
            .setDefault(CcToolchainProvider.class, ccToolchain)
            .setDefault(BuildConfiguration.class, ruleContext.getConfiguration())
            .setDefault(FdoContext.class, fdoContext)
            .setDefault(Label.class, ruleContext.getLabel())
            .setDefault(Artifact.class, grepIncludes)
            .setDefault(CcCompilationOutputs.class, CcCompilationOutputs.builder().build());
    tester.testConstructors(CcCompilationHelper.class, Visibility.PACKAGE);
    tester.testAllPublicInstanceMethods(
        new CcCompilationHelper(
            ruleContext,
            ruleContext,
            target.getLabel(),
            /* grepIncludes= */ null,
            MockCppSemantics.INSTANCE,
            FeatureConfiguration.EMPTY,
            ccToolchain,
            fdoContext,
            /* executionInfo= */ ImmutableMap.of(),
            /* shouldProcessHeaders= */ true));
  }

  @Test
  public void testCanIgnoreObjcSource() throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget("a", "b", "cc_library(name = 'b', srcs = ['cpp.cc'])");
    Artifact objcSrc = getSourceArtifact("objc.m");
    RuleContext ruleContext = getRuleContext(target);
    CcToolchainProvider ccToolchain =
        CppHelper.getToolchainUsingDefaultCcToolchainAttribute(getRuleContext(target));
    FdoContext fdoContext = ccToolchain.getFdoContext();
    CcCompilationHelper helper =
        new CcCompilationHelper(
                ruleContext,
                ruleContext,
                ruleContext.getLabel(),
                /* grepIncludes= */ null,
                MockCppSemantics.INSTANCE,
                FeatureConfiguration.EMPTY,
                ccToolchain,
                fdoContext,
                /* executionInfo= */ ImmutableMap.of(),
                /* shouldProcessHeaders= */ true)
            .addSources(objcSrc);

    ImmutableList.Builder<Artifact> helperArtifacts = ImmutableList.builder();
    for (CppSource source : helper.getCompilationUnitSources()) {
      helperArtifacts.add(source.getSource());
    }

    assertThat(Artifact.toRootRelativePaths(helperArtifacts.build())).isEmpty();
  }

  @Test
  public void testCanConsumeObjcSource() throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget("a", "b", "cc_library(name = 'b', srcs = ['cpp.cc'])");
    Artifact objcSrc = getSourceArtifact("objc.m");
    RuleContext ruleContext = getRuleContext(target);
    CcToolchainProvider ccToolchain =
        CppHelper.getToolchainUsingDefaultCcToolchainAttribute(getRuleContext(target));
    FdoContext fdoContext = ccToolchain.getFdoContext();
    CcCompilationHelper helper =
        new CcCompilationHelper(
                ruleContext,
                ruleContext,
                ruleContext.getLabel(),
                /* grepIncludes= */ null,
                MockCppSemantics.INSTANCE,
                FeatureConfiguration.EMPTY,
                SourceCategory.CC_AND_OBJC,
                ccToolchain,
                fdoContext,
                ruleContext.getConfiguration(),
                ImmutableMap.of(),
                /* shouldProcessHeaders= */ true)
            .addSources(objcSrc);

    ImmutableList.Builder<Artifact> helperArtifacts = ImmutableList.builder();
    for (CppSource source : helper.getCompilationUnitSources()) {
      helperArtifacts.add(source.getSource());
    }

    assertThat(Artifact.toRootRelativePaths(helperArtifacts.build()))
        .contains(objcSrc.getRootRelativePath().toString());
  }

  @Test
  public void testSetAllowCodeCoverage() throws Exception {
    useConfiguration("--collect_code_coverage", "--instrumentation_filter=.");
    ConfiguredTarget target =
        scratchConfiguredTarget("a", "b", "cc_library(name = 'b', srcs = ['cpp.cc'])");
    assertThat(CcCompilationHelper.isCodeCoverageEnabled(getRuleContext(target))).isTrue();
  }
}

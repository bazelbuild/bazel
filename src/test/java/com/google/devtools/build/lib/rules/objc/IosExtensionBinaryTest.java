// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.rules.objc.BinaryLinkingTargetFactory.REQUIRES_AT_LEAST_ONE_LIBRARY_OR_SOURCE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.ASSET_CATALOG;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for ios_extension_binary. */
@RunWith(JUnit4.class)
public class IosExtensionBinaryTest extends ObjcRuleTestCase {
  static final RuleType RULE_TYPE = new OnlyNeedsSourcesRuleType("ios_extension_binary");
  protected static final ExtraLinkArgs EXTRA_LINK_ARGS =
      new ExtraLinkArgs("-e", "_NSExtensionMain", "-fapplication-extension");

  @Before
  public final void initializeToolsConfigMock() throws Exception {
    MockProtoSupport.setup(mockToolsConfig);
    MockObjcSupport.setup(mockToolsConfig);
  }

  @Test
  public void testCreate_runfiles() throws Exception {
    scratch.file("x/a.m");
    RULE_TYPE.scratchTarget(scratch, "srcs", "['a.m']");
    ConfiguredTarget binary = getConfiguredTarget("//x:x");
    RunfilesProvider runfiles = binary.getProvider(RunfilesProvider.class);
    assertThat(runfiles.getDefaultRunfiles().getArtifacts()).isEmpty();
    assertThat(Artifact.toRootRelativePaths(runfiles.getDataRunfiles().getArtifacts()))
        .containsExactly("x/x_bin");
  }

  @Test
  public void testCreate_errorForNoSourceOrDep() throws Exception {
    checkError("x", "x", REQUIRES_AT_LEAST_ONE_LIBRARY_OR_SOURCE_FILE,
        "ios_extension_binary(name='x')");
  }

  @Test
  public void testCompileWithDotMFileInHeaders() throws Exception {
    checkCompileWithDotMFileInHeaders(RULE_TYPE);
  }

  @Test
  public void testObjcProviderExportsAssetCatalogs() throws Exception {
    scratch.file("x/BUILD",
        "ios_extension_binary(",
        "    name = 'x',",
        "    srcs = ['a.m'],",
        "    asset_catalogs = ['foo.xcassets/bar', 'foo.xcassets/baz'],",
        ")");
    ObjcProvider provider = providerForTarget("//x:x");
    assertThat(provider.get(ASSET_CATALOG))
        .containsExactly(
            getSourceArtifact("x/foo.xcassets/bar"),
            getSourceArtifact("x/foo.xcassets/baz"));
  }

  @Test
  public void testLinksFrameworksOfSelfAndTransitiveDependencies() throws Exception {
    checkLinksFrameworksOfSelfAndTransitiveDependencies(RULE_TYPE);
  }

  @Test
  public void testLinksWeakFrameworksOfSelfAndTransitiveDependencies() throws Exception {
    checkLinksWeakFrameworksOfSelfAndTransitiveDependencies(RULE_TYPE);
  }

  @Test
  public void testLinksDylibsTransitively() throws Exception {
    checkLinksDylibsTransitively(RULE_TYPE);
  }

  @Test
  public void testPopulatesCompilationArtifacts() throws Exception {
    checkPopulatesCompilationArtifacts(RULE_TYPE);
  }

  @Test
  public void testArchivesPrecompiledObjectFiles() throws Exception {
    checkArchivesPrecompiledObjectFiles(RULE_TYPE);
  }

  @Test
  public void testErrorsWrongFileTypeForSrcsWhenCompiling() throws Exception {
    checkErrorsWrongFileTypeForSrcsWhenCompiling(RULE_TYPE);
  }

  @Test
  public void testObjcCopts() throws Exception {
    checkObjcCopts(RULE_TYPE);
  }

  @Test
  public void testObjcCopts_argumentOrdering() throws Exception {
    checkObjcCopts_argumentOrdering(RULE_TYPE);
  }

  @Test
  public void testAllowVariousNonBlacklistedTypesInHeaders() throws Exception {
    checkAllowVariousNonBlacklistedTypesInHeaders(RULE_TYPE);
  }

  @Test
  public void testWarningForBlacklistedTypesInHeaders() throws Exception {
    checkWarningForBlacklistedTypesInHeaders(RULE_TYPE);
  }

  @Test
  public void testCppSourceCompilesWithCppFlags() throws Exception {
    checkCppSourceCompilesWithCppFlags(RULE_TYPE);
  }

  @Test
  public void testLinkOpts() throws Exception {
    checkLinkopts(RULE_TYPE);
  }

  @Test
  public void testProtoBundlingAndLinking() throws Exception {
    checkProtoBundlingAndLinking(RULE_TYPE);
  }

  @Test
  public void testProtoBundlingWithTargetsWithNoDeps() throws Exception {
    checkProtoBundlingWithTargetsWithNoDeps(RULE_TYPE);
  }

  @Test
  public void testLinkingRuleCanUseCrosstool() throws Exception {
    checkLinkingRuleCanUseCrosstool(RULE_TYPE);
  }

  @Test
  public void testBinaryStrippings() throws Exception {
    checkBinaryStripAction(RULE_TYPE);
  }

  @Test
  public void testCompilationActionsForDebug() throws Exception {
    checkClangCoptsForCompilationMode(RULE_TYPE, CompilationMode.DBG, CodeCoverageMode.NONE);
  }

  @Test
  public void testCompilationActionsForOptimized() throws Exception {
    checkClangCoptsForCompilationMode(RULE_TYPE, CompilationMode.OPT, CodeCoverageMode.NONE);
  }

  @Test
  public void testClangCoptsForDebugModeWithoutGlib() throws Exception {
    checkClangCoptsForDebugModeWithoutGlib(RULE_TYPE);
  }

  @Test
  public void testLinkActionCorrect() throws Exception {
    checkLinkActionCorrect(RULE_TYPE, EXTRA_LINK_ARGS);
  }

  @Test
  public void testFrameworkDepLinkFlags() throws Exception {
    checkFrameworkDepLinkFlags(RULE_TYPE, EXTRA_LINK_ARGS);
  }

  @Test
  public void testCompilationActionsForDebugInGcovCoverage() throws Exception {
    checkClangCoptsForCompilationMode(RULE_TYPE, CompilationMode.DBG,
        CodeCoverageMode.GCOV);
  }

  @Test
  public void testCompilationActionsForDebugInLlvmCovCoverage() throws Exception {
    checkClangCoptsForCompilationMode(RULE_TYPE, CompilationMode.DBG,
        CodeCoverageMode.LLVMCOV);
  }

  @Test
  public void testCompilationActionsForOptimizedInGcovCoverage() throws Exception {
    checkClangCoptsForCompilationMode(RULE_TYPE, CompilationMode.OPT,
        CodeCoverageMode.GCOV);
  }

  @Test
  public void testCompilationActionsForOptimizedInLlvmCovCoverage() throws Exception {
    checkClangCoptsForCompilationMode(RULE_TYPE, CompilationMode.OPT,
        CodeCoverageMode.LLVMCOV);
  }

  @Test
  public void testCompileWithTextualHeaders() throws Exception {
    checkCompileWithTextualHeaders(RULE_TYPE);
  }

  @Test
  public void testCompilesWithHdrs() throws Exception {
    checkCompilesWithHdrs(RULE_TYPE);
  }

  @Test
  public void testCompilesSources() throws Exception {
    checkCompilesSources(RULE_TYPE);
  }

  @Test
  public void testLinkActionWithTransitiveCppDependency() throws Exception {
    checkLinkActionWithTransitiveCppDependency(RULE_TYPE, EXTRA_LINK_ARGS);
  }
}

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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.rules.cpp.CcCommon.CoptsFilter;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.List;
import java.util.regex.Pattern;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link com.google.devtools.build.lib.rules.cpp.CompileCommandLine}, for example testing
 * the ordering of individual command line flags, or that command line is emitted differently
 * subject to the presence of certain build variables. Also used to test migration logic (removing
 * hardcoded flags and expressing them using feature configuration.
 */
@RunWith(JUnit4.class)
public class CompileCommandLineTest extends BuildViewTestCase {

  private RuleContext ruleContext;

  @Before
  public void initializeRuleContext() throws Exception {
    scratch.file("foo/BUILD", "cc_library(name = 'foo')");
    ruleContext = getRuleContext(getConfiguredTarget("//foo:foo"));
  }

  private Artifact scratchArtifact(String s) {
    Path execRoot = outputBase.getRelative("exec");
    String outSegment = "root";
    Path outputRoot = execRoot.getRelative(outSegment);
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(execRoot, outSegment);
    try {
      return ActionsTestUtil.createArtifact(
          root, scratch.overwriteFile(outputRoot.getRelative(s).toString()));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private static FeatureConfiguration getMockFeatureConfiguration(
      RuleContext ruleContext, String... crosstool) throws Exception {
    return CcToolchainFeaturesTest.buildFeatures(crosstool)
        .getFeatureConfiguration(
            ImmutableSet.of(
                CppActionNames.ASSEMBLE,
                CppActionNames.PREPROCESS_ASSEMBLE,
                CppActionNames.C_COMPILE,
                CppActionNames.CPP_COMPILE,
                CppActionNames.CPP_HEADER_PARSING,
                CppActionNames.CPP_MODULE_CODEGEN,
                CppActionNames.CPP_MODULE_COMPILE));
  }

  @Test
  public void testFeatureConfigurationCommandLineIsUsed() throws Exception {
    CompileCommandLine compileCommandLine =
        makeCompileCommandLineBuilder()
            .setFeatureConfiguration(
                getMockFeatureConfiguration(
                    ruleContext,
                    "",
                    "action_config {",
                    "  config_name: 'c++-compile'",
                    "  action_name: 'c++-compile'",
                    "  implies: 'some_foo_feature'",
                    "  tool {",
                    "    tool_path: 'foo/bar/DUMMY_COMPILER'",
                    "  }",
                    "}",
                    "feature {",
                    "  name: 'some_foo_feature'",
                    "  flag_set {",
                    "     action: 'c++-compile'",
                    "     flag_group {",
                    "       flag: '-some_foo_flag'",
                    "    }",
                    "  }",
                    "}"))
            .build();
    assertThat(
            compileCommandLine.getArguments(
                /* parameterFilePath= */ null, /* overwrittenVariables= */ null))
        .contains("-some_foo_flag");
  }

  @Test
  public void testUnfilteredFlagsAreNotFiltered() throws Exception {
    List<String> actualCommandLine =
        getCompileCommandLineWithCoptsFilter(CppRuleClasses.UNFILTERED_COMPILE_FLAGS_FEATURE_NAME);
    assertThat(actualCommandLine).contains("-i_am_a_flag");
  }

  @Test
  public void testNonUnfilteredFlagsAreFiltered() throws Exception {
    List<String> actualCommandLine = getCompileCommandLineWithCoptsFilter("filtered_flags");
    assertThat(actualCommandLine).doesNotContain("-i_am_a_flag");
  }

  private List<String> getCompileCommandLineWithCoptsFilter(String featureName) throws Exception {
    CompileCommandLine compileCommandLine =
        makeCompileCommandLineBuilder()
            .setFeatureConfiguration(
                getMockFeatureConfiguration(
                    ruleContext,
                    "",
                    "action_config {",
                    "  config_name: 'c++-compile'",
                    "  action_name: 'c++-compile'",
                    "  implies: '" + featureName + "'",
                    "  tool {",
                    "    tool_path: 'foo/bar/DUMMY_COMPILER'",
                    "  }",
                    "}",
                    "feature {",
                    "  name: '" + featureName + "'",
                    "  flag_set {",
                    "     action: 'c++-compile'",
                    "     flag_group {",
                    "       flag: '-i_am_a_flag'",
                    "    }",
                    "  }",
                    "}"))
            .setCoptsFilter(CoptsFilter.fromRegex(Pattern.compile(".*i_am_a_flag.*")))
            .build();
    return compileCommandLine.getArguments(
        /* parameterFilePath= */ null, /* overwrittenVariables= */ null);
  }

  private CompileCommandLine.Builder makeCompileCommandLineBuilder() throws Exception {
    return CompileCommandLine.builder(
        scratchArtifact("a/FakeInput"),
        CoptsFilter.alwaysPasses(),
        "c++-compile",
        scratchArtifact("a/dotD"));
  }
}

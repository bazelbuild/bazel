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

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CompileCommandLine.Builder;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction.DotdFile;
import java.io.IOException;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link CompileCommandLine}, for example testing the ordering of individual command
 * line flags, or that command line is emitted differently subject to the presence of certain
 * build variables. Also used to test migration logic (removing hardcoded flags and expressing
 * them using feature configuration.
 */
@RunWith(JUnit4.class)
public class CompileCommandLineTest extends BuildViewTestCase {

  private Artifact scratchArtifact(String s) {
    try {
      return new Artifact(
          scratch.overwriteFile(
              outputBase.getRelative("compile_command_line").getRelative(s).toString()),
          Root.asDerivedRoot(
              scratch.dir(outputBase.getRelative("compile_command_line").toString())));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private static FeatureConfiguration getMockFeatureConfiguration(String... crosstool)
      throws Exception {
    return CcToolchainFeaturesTest.buildFeatures(crosstool)
        .getFeatureConfiguration(
            FeatureSpecification.create(
                ImmutableSet.of(
                    CppCompileAction.ASSEMBLE,
                    CppCompileAction.PREPROCESS_ASSEMBLE,
                    CppCompileAction.C_COMPILE,
                    CppCompileAction.CPP_COMPILE,
                    CppCompileAction.CPP_HEADER_PARSING,
                    CppCompileAction.CPP_HEADER_PREPROCESSING,
                    CppCompileAction.CPP_MODULE_CODEGEN,
                    CppCompileAction.CPP_MODULE_COMPILE),
                ImmutableSet.<String>of()));
  }

  @Test
  public void testFeatureConfigurationCommandLineIsUsed() throws Exception {
    CompileCommandLine compileCommandLine =
        makeCompileCommandLineBuilder()
            .setFeatureConfiguration(
                getMockFeatureConfiguration(
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
    assertThat(compileCommandLine.getArgv(scratchArtifact("a/FakeOutput").getExecPath(), null))
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
            .setCoptsFilter(flag -> !flag.contains("i_am_a_flag"))
            .build();
    return compileCommandLine.getArgv(scratchArtifact("a/FakeOutput").getExecPath(), null);
  }

  private Builder makeCompileCommandLineBuilder() throws Exception {
    ConfiguredTarget dummyTarget =
        scratchConfiguredTarget("a", "a", "cc_binary(name='a', srcs=['a.cc'])");
    return CompileCommandLine.builder(
        scratchArtifact("a/FakeInput"),
        scratchArtifact("a/FakeOutput"),
        new Predicate<String>() {
          @Override
          public boolean apply(String s) {
            return true;
          }
        },
        "c++-compile",
        getTargetConfiguration().getFragment(CppConfiguration.class),
        new DotdFile(scratchArtifact("a/dotD")),
        CppHelper.getToolchainUsingDefaultCcToolchainAttribute(getRuleContext(dummyTarget)));
  }
}

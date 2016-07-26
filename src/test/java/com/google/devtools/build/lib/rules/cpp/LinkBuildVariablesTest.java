// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests that {@code CppLinkAction} is populated with the correct build variables. */
@RunWith(JUnit4.class)
public class LinkBuildVariablesTest extends BuildViewTestCase {

  private CppLinkAction getCppLinkAction(ConfiguredTarget target, Link.LinkTargetType type) {
    Artifact linkerOutput = null;
    switch (type) {
      case STATIC_LIBRARY:
      case ALWAYS_LINK_STATIC_LIBRARY:
        linkerOutput = getBinArtifact("lib" + target.getLabel().getName() + ".a", target);
        break;
      case PIC_STATIC_LIBRARY:
      case ALWAYS_LINK_PIC_STATIC_LIBRARY:
        linkerOutput = getBinArtifact("lib" + target.getLabel().getName() + "pic.a", target);
        break;
      case DYNAMIC_LIBRARY:
        linkerOutput = getBinArtifact("lib" + target.getLabel().getName() + ".so", target);
        break;
      case EXECUTABLE:
        linkerOutput = getExecutable(target);
        break;
      default:
        throw new IllegalArgumentException(
            String.format("Cannot get CppLinkAction for link type %s", type));
    }
    return (CppLinkAction) getGeneratingAction(linkerOutput);
  }

  private Variables getLinkBuildVariables(ConfiguredTarget target, Link.LinkTargetType type) {
    return getCppLinkAction(target, type).getLinkCommandLine().getBuildVariables();
  }

  private List<String> getVariableValue(Variables variables, String variable) throws Exception {
    FeatureConfiguration mockFeatureConfiguration =
        CcToolchainFeaturesTest.buildFeatures(
                "feature {",
                "   name: 'a'",
                "   flag_set {",
                "   action: 'foo'",
                "      flag_group {",
                "         flag: '%{" + variable + "}'",
                "      }",
                "   }",
                "}")
            .getFeatureConfiguration("a");
    return mockFeatureConfiguration.getCommandLine("foo", variables);
  }

  @Test
  public void testLinkstampBuildVariable() throws Exception {
    scratch.file(
        "x/BUILD",
        "cc_binary(",
        "   name = 'bin',",
        "   srcs = ['a.cc'],",
        "   deps = [':lib'],",
        ")",
        "cc_library(",
        "   name = 'lib',",
        "   srcs = ['b.cc'],",
        "   linkstamp = 'c.cc',",
        ")");
    scratch.file("x/a.cc");
    scratch.file("x/b.cc");
    scratch.file("x/c.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:bin");
    Variables variables = getLinkBuildVariables(target, Link.LinkTargetType.EXECUTABLE);
    List<String> variableValue =
        getVariableValue(variables, CppLinkActionBuilder.LINKSTAMP_PATHS_VARIABLE);
    assertThat(Iterables.getOnlyElement(variableValue)).contains("c.o");
  }

  @Test
  public void testForcePicBuildVariable() throws Exception {
    useConfiguration("--force_pic");
    scratch.file("x/BUILD", "cc_binary(", "   name = 'bin',", "   srcs = ['a.cc'],", ")");
    scratch.file("x/a.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:bin");
    Variables variables = getLinkBuildVariables(target, Link.LinkTargetType.EXECUTABLE);
    List<String> variableValue =
        getVariableValue(variables, CppLinkActionBuilder.FORCE_PIC_VARIABLE);
    assertThat(variableValue).contains("");
  }

  @Test
  public void testWholeArchiveBuildVariables() throws Exception {
    scratch.file(
        "x/BUILD",
        "cc_binary(",
        "   name = 'bin.so',",
        "   srcs = ['a.cc'],",
        "   linkopts = ['-shared'],",
        "   linkstatic = 1",
        ")");
    scratch.file("x/a.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:bin.so");
    Variables variables = getLinkBuildVariables(target, Link.LinkTargetType.EXECUTABLE);
    List<String> variableValue =
        getVariableValue(variables, CppLinkActionBuilder.GLOBAL_WHOLE_ARCHIVE_VARIABLE);
    assertThat(variableValue).contains("");
  }
}

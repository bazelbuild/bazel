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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.protobuf.TextFormat;
import java.util.List;

/**
 * Common test code to test that {@code CppLinkAction} is populated with the correct build
 * variables.
 **/
public class LinkBuildVariablesTestCase extends BuildViewTestCase {

  protected CppLinkAction getCppLinkAction(ConfiguredTarget target, Link.LinkTargetType type) {
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
      case NODEPS_DYNAMIC_LIBRARY:
        linkerOutput = getBinArtifact("lib" + target.getLabel().getName() + ".so", target);
        break;
      case DYNAMIC_LIBRARY:
        linkerOutput = getBinArtifact(target.getLabel().getName(), target);
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

  /** Returns active build variables for a link action of given type for given target. */
  protected CcToolchainVariables getLinkBuildVariables(
      ConfiguredTarget target, Link.LinkTargetType type) {
    return getCppLinkAction(target, type).getLinkCommandLineForTesting().getBuildVariables();
  }

  /** Creates a CcToolchainFeatures from features described in the given toolchain fragment. */
  public static CcToolchainFeatures buildFeatures(RuleContext ruleContext, String... toolchain)
      throws Exception {
    CToolchain.Builder toolchainBuilder = CToolchain.newBuilder();
    TextFormat.merge(Joiner.on("").join(toolchain), toolchainBuilder);
    return new CcToolchainFeatures(
        CcToolchainConfigInfo.fromToolchainForTestingOnly(toolchainBuilder.buildPartial()),
        /* ccToolchainPath= */ PathFragment.EMPTY_FRAGMENT);
  }

  /** Returns the value of a given sequence variable in context of the given Variables instance. */
  protected static List<String> getSequenceVariableValue(
      RuleContext ruleContext, CcToolchainVariables variables, String variable) throws Exception {
    FeatureConfiguration mockFeatureConfiguration =
        buildFeatures(
                ruleContext,
                "feature {",
                "  name: 'a'",
                "  flag_set {",
                "  action: 'foo'",
                "    flag_group {",
                "      iterate_over: '" + variable + "'",
                "      flag: '%{" + variable + "}'",
                "    }",
                "  }",
                "}")
            .getFeatureConfiguration(ImmutableSet.of("a"));
    return mockFeatureConfiguration.getCommandLine("foo", variables);
  }

  /** Returns the value of a given string variable in context of the given Variables instance. */
  protected static String getVariableValue(
      RuleContext ruleContext, CcToolchainVariables variables, String variable) throws Exception {
    FeatureConfiguration mockFeatureConfiguration =
        buildFeatures(
                ruleContext,
                "feature {",
                "  name: 'a'",
                "  flag_set {",
                "  action: 'foo'",
                "    flag_group {",
                "      flag: '%{" + variable + "}'",
                "    }",
                "  }",
                "}")
            .getFeatureConfiguration(ImmutableSet.of("a"));
    return Iterables.getOnlyElement(mockFeatureConfiguration.getCommandLine("foo", variables));
  }
}

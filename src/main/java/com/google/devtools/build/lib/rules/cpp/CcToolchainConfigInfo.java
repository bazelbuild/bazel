// Copyright 2018 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.rules.cpp.CcToolchainFeaturesLib.actionConfigFromStarlark;
import static com.google.devtools.build.lib.rules.cpp.CcToolchainFeaturesLib.featureFromStarlark;
import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuild;
import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuiltins;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkProviderWrapper;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ActionConfig;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ArtifactNamePatternMapper;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Feature;
import com.google.devtools.build.lib.util.OS;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;

/** Information describing C++ toolchain derived from CROSSTOOL file. */
@Immutable
public class CcToolchainConfigInfo {

  /** Singleton provider instance for {@link CcToolchainConfigInfo}. */
  public static final Provider PROVIDER = new Provider();

  public static final RulesCcProvider RULES_CC_PROVIDER = new RulesCcProvider();

  public static CcToolchainConfigInfo get(ConfiguredTarget target) throws RuleErrorException {
    CcToolchainConfigInfo info = target.get(PROVIDER);
    if (info == null) {
      info = target.get(RULES_CC_PROVIDER);
    }
    return info;
  }

  private final StarlarkInfo actual;

  CcToolchainConfigInfo(StarlarkInfo actual) {
    this.actual = actual;
  }

  public ImmutableList<ActionConfig> getActionConfigs() throws EvalException {
    OS execOs = OS.valueOf(actual.getValue("_exec_os_DO_NOT_USE", String.class));
    ImmutableList.Builder<ActionConfig> actionConfigBuilder = ImmutableList.builder();
    for (StarlarkInfo actionConfig :
        Sequence.cast(
            actual.getValue("_action_configs_DO_NOT_USE"), StarlarkInfo.class, "_action_configs")) {
      actionConfigBuilder.add(actionConfigFromStarlark(actionConfig, execOs));
    }
    return actionConfigBuilder.build();
  }

  public ImmutableList<Feature> getFeatures() throws EvalException {
    ImmutableList.Builder<Feature> featureBuilder = ImmutableList.builder();
    for (StarlarkInfo feature :
        Sequence.cast(actual.getValue("_features_DO_NOT_USE"), StarlarkInfo.class, "_features")) {
      featureBuilder.add(featureFromStarlark(feature));
    }
    return featureBuilder.build();
  }

  public ArtifactNamePatternMapper getArtifactNamePatterns() throws EvalException {
    CcToolchainFeatures.ArtifactNamePatternMapper.Builder artifactNamePatternBuilder =
        new CcToolchainFeatures.ArtifactNamePatternMapper.Builder();
    for (StarlarkInfo artifactNamePattern :
        Sequence.cast(
            actual.getValue("_artifact_name_patterns_DO_NOT_USE"),
            StarlarkInfo.class,
            "_artifact_name_patterns")) {
      CcToolchainFeaturesLib.artifactNamePatternFromStarlark(
          artifactNamePattern, artifactNamePatternBuilder::addOverride);
    }
    return artifactNamePatternBuilder.build();
  }

  /** Provider class for {@link CcToolchainConfigInfo} objects. */
  public static class Provider extends StarlarkProviderWrapper<CcToolchainConfigInfo> {

    private Provider() {
      super(
          keyForBuiltins(
              Label.parseCanonicalUnchecked(
                  "@_builtins//:common/cc/toolchain_config/cc_toolchain_config_info.bzl")),
          "CcToolchainConfigInfo");
    }

    @Override
    public CcToolchainConfigInfo wrap(Info value) throws RuleErrorException {
      return new CcToolchainConfigInfo((StarlarkInfo) value);
    }
  }

  /** Provider class for {@link CcToolchainConfigInfo} objects. */
  public static class RulesCcProvider extends StarlarkProviderWrapper<CcToolchainConfigInfo> {

    private RulesCcProvider() {
      super(
          keyForBuild(
              Label.parseCanonicalUnchecked(
                  "@rules_cc+//cc/private/toolchain_config:cc_toolchain_config_info.bzl")),
          "CcToolchainConfigInfo");
    }

    @Override
    public CcToolchainConfigInfo wrap(Info value) throws RuleErrorException {
      return new CcToolchainConfigInfo((StarlarkInfo) value);
    }
  }
}

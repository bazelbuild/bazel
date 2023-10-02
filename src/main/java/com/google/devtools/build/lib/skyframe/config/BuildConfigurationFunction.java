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
package com.google.devtools.build.lib.skyframe.config;

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationValueEvent;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentFactory;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.transitions.BaselineOptionsValue;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.WorkspaceNameValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;

/** A builder for {@link BuildConfigurationValue} instances. */
public final class BuildConfigurationFunction implements SkyFunction {

  private final BlazeDirectories directories;
  private final ConfiguredRuleClassProvider ruleClassProvider;
  private final FragmentFactory fragmentFactory = new FragmentFactory();

  public BuildConfigurationFunction(
      BlazeDirectories directories, RuleClassProvider ruleClassProvider) {
    this.directories = directories;
    this.ruleClassProvider = (ConfiguredRuleClassProvider) ruleClassProvider;
  }

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, BuildConfigurationFunctionException {
    WorkspaceNameValue workspaceNameValue = (WorkspaceNameValue) env
        .getValue(WorkspaceNameValue.key());
    if (workspaceNameValue == null) {
      return null;
    }

    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }
    BuildConfigurationKey key = (BuildConfigurationKey) skyKey.argument();

    BuildOptions targetOptions = key.getOptions();
    CoreOptions coreOptions = targetOptions.get(CoreOptions.class);

    BuildOptions baselineOptions = null;
    if (coreOptions.useBaselineForOutputDirectoryNamingScheme()) {
      boolean applyExecTransitionToBaseline =
          coreOptions.outputDirectoryNamingScheme.equals(
                  CoreOptions.OutputDirectoryNamingScheme.DIFF_AGAINST_DYNAMIC_BASELINE)
              && coreOptions.isExec;
      var baselineOptionsValue =
          (BaselineOptionsValue)
              env.getValue(BaselineOptionsValue.key(applyExecTransitionToBaseline));
      if (baselineOptionsValue == null) {
        return null;
      }
      baselineOptions = baselineOptionsValue.toOptions();
    }

    try {
      var configurationValue =
          BuildConfigurationValue.create(
              targetOptions,
              baselineOptions,
              workspaceNameValue.getName(),
              starlarkSemantics.getBool(
                  BuildLanguageOptions.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT),
              // Arguments below this are server-global.
              directories,
              ruleClassProvider,
              fragmentFactory);
      env.getListener().post(ConfigurationValueEvent.create(configurationValue));
      return configurationValue;
    } catch (InvalidConfigurationException e) {
      throw new BuildConfigurationFunctionException(e);
    }
  }

  private static final class BuildConfigurationFunctionException extends SkyFunctionException {
    BuildConfigurationFunctionException(Exception e) {
      super(e, Transience.PERSISTENT);
    }
  }
}

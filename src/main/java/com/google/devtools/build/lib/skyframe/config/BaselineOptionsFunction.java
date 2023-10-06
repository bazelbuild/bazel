// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.analysis.config.transitions.BaselineOptionsValue;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionUtil;
import com.google.devtools.build.lib.analysis.producers.BuildConfigurationKeyProducer;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.Driver;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.common.options.OptionsParsingException;
import javax.annotation.Nullable;

/** A builder for {@link BaselineOptionsValue} instances. */
public final class BaselineOptionsFunction implements SkyFunction {
  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, BaselineOptionsFunctionException {
    BaselineOptionsValue.Key key = (BaselineOptionsValue.Key) skyKey.argument();

    BuildOptions rawBaselineOptions = PrecomputedValue.BASELINE_CONFIGURATION.get(env);

    // Some test infrastructure only creates mock or partial top-level BuildOptions such that
    // PlatformOptions or even CoreOptions might not be included.
    // In that case, is not worth doing any special processing of the baseline.
    if (rawBaselineOptions.hasNoConfig()) {
      return BaselineOptionsValue.create(rawBaselineOptions);
    }

    BuildOptions mappedBaselineOptions = mapBuildOptions(env, rawBaselineOptions);
    if (mappedBaselineOptions == null) {
      return null;
    }

    if (key.afterExecTransition()) {
      // A null executionPlatform actually skips transition application so need some value here.
      // It is safe to supply some fake value here (as long as it is constant) since the baseline
      // should never be used to actually construct an action or do toolchain resolution
      // TODO(twigg): This can eventually be replaced by the actual exec platform once
      //   platforms is explicitly in the output path (with the garbage value as a fallback).
      PatchTransition execTransition =
          ExecutionTransitionFactory.createTransition(
              Label.parseCanonicalUnchecked("//this_is_a_faked_exec_platform_for_blaze_internals"));
      BuildOptions toOptions =
          execTransition.patch(
              TransitionUtil.restrict(execTransition, mappedBaselineOptions), env.getListener());
      return BaselineOptionsValue.create(toOptions);
    } else {
      return BaselineOptionsValue.create(mappedBaselineOptions);
    }
  }

  @Nullable
  private static BuildOptions mapBuildOptions(Environment env, BuildOptions rawBaselineOptions)
      throws InterruptedException, BaselineOptionsFunctionException {
    BuildOptionsMapper mapper = new BuildOptionsMapper(rawBaselineOptions);

    try {
      BuildConfigurationKey key = mapper.drive(env);
      if (key == null) {
        return null;
      }
      return key.getOptions();
    } catch (OptionsParsingException e) {
      throw new BaselineOptionsFunctionException(e);
    }
  }

  /** Uses BuildConfigurationKeyProducer to handle finalizing the options. */
  private static class BuildOptionsMapper implements BuildConfigurationKeyProducer.ResultSink {
    private static final String TRANSITION_KEY = "key";

    private final Driver driver;
    private ImmutableMap<String, BuildConfigurationKey> transitionedOptions;
    private OptionsParsingException transitionError;

    private BuildOptionsMapper(BuildOptions options) {
      this.driver =
          new Driver(
              new BuildConfigurationKeyProducer(
                  this, StateMachine.DONE, ImmutableMap.of(TRANSITION_KEY, options)));
    }

    @Override
    public void acceptTransitionError(OptionsParsingException e) {
      this.transitionError = e;
    }

    @Override
    public void acceptTransitionedConfigurations(
        ImmutableMap<String, BuildConfigurationKey> transitionedOptions) {
      this.transitionedOptions = transitionedOptions;
    }

    @Nullable
    private BuildConfigurationKey drive(LookupEnvironment env)
        throws OptionsParsingException, InterruptedException {
      if (!this.driver.drive(env)) {
        return null;
      }

      if (this.transitionError != null) {
        throw this.transitionError;
      }

      return this.transitionedOptions.get(TRANSITION_KEY);
    }
  }

  private static final class BaselineOptionsFunctionException extends SkyFunctionException {
    BaselineOptionsFunctionException(Exception e) {
      super(e, Transience.PERSISTENT);
    }
  }
}

// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.analysis.Aspect;
import com.google.devtools.build.lib.analysis.CachingAnalysisEnvironment;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.AspectFactory;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.AspectValue.AspectKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction.DependencyEvaluationException;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor.BuildViewProvider;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.Set;

import javax.annotation.Nullable;

/**
 * The Skyframe function that generates aspects.
 */
public final class AspectFunction implements SkyFunction {
  private final BuildViewProvider buildViewProvider;

  public AspectFunction(BuildViewProvider buildViewProvider) {
    this.buildViewProvider = buildViewProvider;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws AspectFunctionException {
    SkyframeBuildView view = buildViewProvider.getSkyframeBuildView();
    AspectKey key = (AspectKey) skyKey.argument();
    ConfiguredAspectFactory aspectFactory =
        (ConfiguredAspectFactory) AspectFactory.Util.create(key.getAspect());

    PackageValue packageValue =
        (PackageValue) env.getValue(PackageValue.key(key.getLabel().getPackageIdentifier()));
    if (packageValue == null) {
      return null;
    }

    Target target;
    try {
      target = packageValue.getPackage().getTarget(key.getLabel().getName());
    } catch (NoSuchTargetException e) {
      throw new AspectFunctionException(skyKey, e);
    }

    if (!(target instanceof Rule)) {
      throw new AspectFunctionException(new AspectCreationException(
          "aspects must be attached to rules"));
    }

    final ConfiguredTargetValue configuredTargetValue =
        (ConfiguredTargetValue)
            env.getValue(ConfiguredTargetValue.key(key.getLabel(), key.getConfiguration()));
    if (configuredTargetValue == null) {
      return null;
    }
    RuleConfiguredTarget associatedTarget =
        (RuleConfiguredTarget) configuredTargetValue.getConfiguredTarget();

    if (associatedTarget == null) {
      return null;
    }

    SkyframeDependencyResolver resolver = view.createDependencyResolver(env);
    if (resolver == null) {
      return null;
    }

    TargetAndConfiguration ctgValue =
        new TargetAndConfiguration(target, key.getConfiguration());

    try {
      // Get the configuration targets that trigger this rule's configurable attributes.
      Set<ConfigMatchingProvider> configConditions =
          ConfiguredTargetFunction.getConfigConditions(target, env, resolver, ctgValue);
      if (configConditions == null) {
        // Those targets haven't yet been resolved.
        return null;
      }

      ListMultimap<Attribute, ConfiguredTarget> depValueMap =
          ConfiguredTargetFunction.computeDependencies(env, resolver, ctgValue,
              aspectFactory.getDefinition(), configConditions);

      return createAspect(env, key, associatedTarget, configConditions, depValueMap);
    } catch (DependencyEvaluationException e) {
      throw new AspectFunctionException(e.getRootCauseSkyKey(), e.getCause());
    }
  }

  @Nullable
  private AspectValue createAspect(Environment env, AspectKey key,
      RuleConfiguredTarget associatedTarget, Set<ConfigMatchingProvider> configConditions,
      ListMultimap<Attribute, ConfiguredTarget> directDeps)
      throws AspectFunctionException {
    SkyframeBuildView view = buildViewProvider.getSkyframeBuildView();
    BuildConfiguration configuration = associatedTarget.getConfiguration();

    StoredEventHandler events = new StoredEventHandler();
    CachingAnalysisEnvironment analysisEnvironment = view.createAnalysisEnvironment(
        key, false, events, env, configuration);
    if (env.valuesMissing()) {
      return null;
    }

    ConfiguredAspectFactory aspectFactory =
        (ConfiguredAspectFactory) AspectFactory.Util.create(key.getAspect());
    Aspect aspect = view.createAspect(
        analysisEnvironment, associatedTarget, aspectFactory, directDeps, configConditions);

    events.replayOn(env.getListener());
    if (events.hasErrors()) {
      analysisEnvironment.disable(associatedTarget.getTarget());
      throw new AspectFunctionException(new AspectCreationException(
          "Analysis of target '" + associatedTarget.getLabel() + "' failed; build aborted"));
    }
    Preconditions.checkState(!analysisEnvironment.hasErrors(),
        "Analysis environment hasError() but no errors reported");

    if (env.valuesMissing()) {
      return null;
    }

    analysisEnvironment.disable(associatedTarget.getTarget());
    Preconditions.checkNotNull(aspect);

    return new AspectValue(
        key,
        associatedTarget.getLabel(),
        associatedTarget.getTarget().getLocation(),
        aspect,
        ImmutableList.copyOf(analysisEnvironment.getRegisteredActions()));
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
  
  /**
   * An exception indicating that there was a problem creating an aspect.
   */
  public static final class AspectCreationException extends Exception {
    public AspectCreationException(String message) {
      super(message);
    }
  }

  /**
   * Used to indicate errors during the computation of an {@link AspectValue}.
   */
  private static final class AspectFunctionException extends SkyFunctionException {
    public AspectFunctionException(Exception e) {
      super(e, Transience.PERSISTENT);
    }

    /** Used to rethrow a child error that we cannot handle. */
    public AspectFunctionException(SkyKey childKey, Exception transitiveError) {
      super(transitiveError, childKey);
    }
  }
}

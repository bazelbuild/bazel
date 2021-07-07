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

package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.analysis.AspectCollection;
import com.google.devtools.build.lib.analysis.AspectCollection.AspectCycleOnPathException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.AspectsListBuilder;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.StarlarkAspect;
import com.google.devtools.build.lib.packages.StarlarkAspectClass;
import com.google.devtools.build.lib.skyframe.AspectValueKey.AspectKey;
import com.google.devtools.build.lib.skyframe.AspectValueKey.TopLevelAspectsKey;
import com.google.devtools.build.lib.skyframe.LoadStarlarkAspectFunction.StarlarkAspectLoadingKey;
import com.google.devtools.build.lib.skyframe.LoadStarlarkAspectFunction.StarlarkAspectLoadingValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/**
 * SkyFunction to load top level aspects, build the dependency relation between them based on the
 * providers they advertise and provide using {@link AspectCollection} and runs the obtained aspects
 * path on the top level target.
 *
 * <p>Used for loading top-level aspects. At top level, in {@link
 * com.google.devtools.build.lib.analysis.BuildView}, we cannot invoke two SkyFunctions one after
 * another, so BuildView calls this function to do the work.
 */
public class ToplevelStarlarkAspectFunction implements SkyFunction {
  ToplevelStarlarkAspectFunction() {}

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws TopLevelStarlarkAspectFunctionException, InterruptedException {
    TopLevelAspectsKey topLevelAspectsKey = (TopLevelAspectsKey) skyKey.argument();

    ImmutableList<Aspect> topLevelAspects;
    try {
      topLevelAspects = getTopLevelAspects(env, topLevelAspectsKey.getTopLevelAspectsClasses());
    } catch (EvalException e) {
      env.getListener().handle(Event.error(e.getMessage()));
      throw new TopLevelStarlarkAspectFunctionException(
          new AspectCreationException(e.getMessage(), topLevelAspectsKey.getLabel()));
    }

    if (topLevelAspects == null) {
      return null; // some aspects are not loaded
    }

    AspectCollection aspectCollection;
    try {
      aspectCollection = AspectCollection.create(topLevelAspects);
    } catch (AspectCycleOnPathException e) {
      env.getListener().handle(Event.error(e.getMessage()));
      throw new TopLevelStarlarkAspectFunctionException(
          new AspectCreationException(e.getMessage(), topLevelAspectsKey.getLabel()));
    }

    ImmutableList<AspectKey> aspectKeys =
        getTopLevelAspectsKeys(aspectCollection, topLevelAspectsKey.getBaseConfiguredTargetKey());

    Map<SkyKey, SkyValue> result = env.getValues(aspectKeys);
    if (env.valuesMissing()) {
      return null; // some aspects keys are not evaluated
    }

    return new TopLevelAspectsValue(result);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  @Nullable
  private static ImmutableList<Aspect> getTopLevelAspects(
      Environment env, ImmutableList<AspectClass> topLevelAspectsClasses)
      throws InterruptedException, EvalException {
    AspectsListBuilder aspectsList = new AspectsListBuilder();

    ImmutableList.Builder<StarlarkAspectLoadingKey> aspectLoadingKeys = ImmutableList.builder();
    for (AspectClass aspectClass : topLevelAspectsClasses) {
      if (aspectClass instanceof StarlarkAspectClass) {
        aspectLoadingKeys.add(
            LoadStarlarkAspectFunction.createStarlarkAspectLoadingKey(
                (StarlarkAspectClass) aspectClass));
      }
    }

    Map<SkyKey, SkyValue> loadedAspects = env.getValues(aspectLoadingKeys.build());
    if (env.valuesMissing()) {
      return null;
    }

    for (AspectClass aspectClass : topLevelAspectsClasses) {
      if (aspectClass instanceof StarlarkAspectClass) {
        StarlarkAspectLoadingValue aspectLoadingValue =
            (StarlarkAspectLoadingValue)
                loadedAspects.get(
                    LoadStarlarkAspectFunction.createStarlarkAspectLoadingKey(
                        (StarlarkAspectClass) aspectClass));
        StarlarkAspect starlarkAspect = aspectLoadingValue.getAspect();
        starlarkAspect.attachToAspectsList(
            /** baseAspectName= */
            null,
            aspectsList,
            /** inheritedRequiredProviders= */
            ImmutableList.of(),
            /** inheritedAttributeAspects= */
            ImmutableList.of());
      } else {
        aspectsList.addAspect((NativeAspectClass) aspectClass);
      }
    }

    return aspectsList.buildAspects();
  }

  private static ImmutableList<AspectKey> getTopLevelAspectsKeys(
      AspectCollection aspectCollection, ConfiguredTargetKey topLevelTargetKey) {
    Map<AspectDescriptor, AspectKey> result = new HashMap<>();
    for (AspectCollection.AspectDeps aspectDeps : aspectCollection.getUsedAspects()) {
      buildAspectKey(aspectDeps, result, topLevelTargetKey);
    }
    return ImmutableList.copyOf(result.values());
  }

  private static AspectKey buildAspectKey(
      AspectCollection.AspectDeps aspectDeps,
      Map<AspectDescriptor, AspectKey> result,
      ConfiguredTargetKey topLevelTargetKey) {
    if (result.containsKey(aspectDeps.getAspect())) {
      return result.get(aspectDeps.getAspect());
    }

    ImmutableList.Builder<AspectKey> dependentAspects = ImmutableList.builder();
    for (AspectCollection.AspectDeps path : aspectDeps.getUsedAspects()) {
      dependentAspects.add(buildAspectKey(path, result, topLevelTargetKey));
    }

    AspectKey aspectKey =
        AspectValueKey.createAspectKey(
            aspectDeps.getAspect(),
            dependentAspects.build(),
            topLevelTargetKey.getConfigurationKey(),
            topLevelTargetKey);
    result.put(aspectKey.getAspectDescriptor(), aspectKey);
    return aspectKey;
  }

  /** Exceptions thrown from ToplevelStarlarkAspectFunction. */
  public static class TopLevelStarlarkAspectFunctionException extends SkyFunctionException {
    public TopLevelStarlarkAspectFunctionException(AspectCreationException cause) {
      super(cause, Transience.PERSISTENT);
    }
  }

  /**
   * SkyValue for {@code TopLevelAspectsKey} wraps a list of the {@code AspectValue} of the top
   * level aspects applied on the same top level target.
   */
  public static class TopLevelAspectsValue implements ActionLookupValue {
    private final Map<SkyKey, SkyValue> topLevelAspectsMap;

    public TopLevelAspectsValue(Map<SkyKey, SkyValue> topLevelAspectsMap) {
      this.topLevelAspectsMap = topLevelAspectsMap;
    }

    public ImmutableList<SkyValue> getTopLevelAspectsValues() {
      return ImmutableList.copyOf(topLevelAspectsMap.values());
    }

    public SkyValue get(SkyKey skyKey) {
      return topLevelAspectsMap.get(skyKey);
    }

    @Override
    public ImmutableList<ActionAnalysisMetadata> getActions() {
      // return topLevelAspectsMap.values().stream().
      return ImmutableList.of();
    }
  }
}

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
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.TopLevelAspectsKey;
import com.google.devtools.build.lib.skyframe.BuildTopLevelAspectsDetailsFunction.AspectDetails;
import com.google.devtools.build.lib.skyframe.BuildTopLevelAspectsDetailsFunction.BuildTopLevelAspectsDetailsKey;
import com.google.devtools.build.lib.skyframe.BuildTopLevelAspectsDetailsFunction.BuildTopLevelAspectsDetailsValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * SkyFunction to run the aspects path obtained from top-level aspects on the list of top-level
 * targets.
 *
 * <p>Used for loading top-level aspects. At top level, in {@link
 * com.google.devtools.build.lib.analysis.BuildView}, we cannot invoke two SkyFunctions one after
 * another, so BuildView calls this function to do the work.
 */
public final class ToplevelStarlarkAspectFunction implements SkyFunction {

  ToplevelStarlarkAspectFunction() {}

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws TopLevelStarlarkAspectFunctionException, InterruptedException {
    TopLevelAspectsKey topLevelAspectsKey = (TopLevelAspectsKey) skyKey.argument();

    BuildTopLevelAspectsDetailsValue topLevelAspectsDetails =
        (BuildTopLevelAspectsDetailsValue)
            env.getValue(
                BuildTopLevelAspectsDetailsKey.create(
                    topLevelAspectsKey.getTopLevelAspectsClasses(),
                    topLevelAspectsKey.getTopLevelAspectsParameters()));
    if (topLevelAspectsDetails == null) {
      return null; // some aspects details are not ready
    }

    Collection<AspectKey> aspectsKeys =
        getTopLevelAspectsKeys(
            topLevelAspectsDetails.getAspectsDetails(),
            topLevelAspectsKey.getBaseConfiguredTargetKey());

    Map<SkyKey, SkyValue> result = env.getValues(aspectsKeys);
    if (env.valuesMissing()) {
      return null; // some aspects keys are not evaluated
    }

    return new TopLevelAspectsValue(result.values());
  }

  private static Collection<AspectKey> getTopLevelAspectsKeys(
      ImmutableList<AspectDetails> aspectsDetails, ConfiguredTargetKey topLevelTargetKey) {
    Map<AspectDescriptor, AspectKey> result = new HashMap<>();
    for (AspectDetails aspect : aspectsDetails) {
      buildAspectKey(aspect, result, topLevelTargetKey);
    }
    return result.values();
  }

  private static AspectKey buildAspectKey(
      AspectDetails aspect,
      Map<AspectDescriptor, AspectKey> result,
      ConfiguredTargetKey topLevelTargetKey) {
    if (result.containsKey(aspect.getAspectDescriptor())) {
      return result.get(aspect.getAspectDescriptor());
    }

    ImmutableList.Builder<AspectKey> dependentAspects = ImmutableList.builder();
    for (AspectDetails depAspect : aspect.getUsedAspects()) {
      dependentAspects.add(buildAspectKey(depAspect, result, topLevelTargetKey));
    }

    AspectKey aspectKey =
        AspectKeyCreator.createAspectKey(
            aspect.getAspectDescriptor(),
            dependentAspects.build(),
            topLevelTargetKey);
    result.put(aspectKey.getAspectDescriptor(), aspectKey);
    return aspectKey;
  }

  /** Exceptions thrown from ToplevelStarlarkAspectFunction. */
  public static final class TopLevelStarlarkAspectFunctionException extends SkyFunctionException {
    public TopLevelStarlarkAspectFunctionException(AspectCreationException cause) {
      super(cause, Transience.PERSISTENT);
    }
  }

  /**
   * SkyValue for {@code TopLevelAspectsKey} wraps a list of the {@code AspectValue} of the top
   * level aspects applied on the same top level target.
   */
  public static class TopLevelAspectsValue implements ActionLookupValue {
    private final ImmutableList<SkyValue> topLevelAspectsValues;

    public TopLevelAspectsValue(Collection<SkyValue> topLevelAspectsValues) {
      this.topLevelAspectsValues = ImmutableList.copyOf(topLevelAspectsValues);
    }

    public ImmutableList<SkyValue> getTopLevelAspectsValues() {
      return topLevelAspectsValues;
    }

    @Override
    public ImmutableList<ActionAnalysisMetadata> getActions() {
      return ImmutableList.of();
    }
  }
}

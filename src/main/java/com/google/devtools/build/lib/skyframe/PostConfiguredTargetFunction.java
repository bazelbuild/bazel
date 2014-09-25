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

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.TargetAndConfiguration;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import javax.annotation.Nullable;

/**
 * Build a post-processed ConfiguredTarget, vetting it for action conflict issues.
 */
public class PostConfiguredTargetFunction implements SkyFunction {
  private static final Function<TargetAndConfiguration, SkyKey> TO_KEYS =
      new Function<TargetAndConfiguration, SkyKey>() {
    @Override
    public SkyKey apply(TargetAndConfiguration input) {
      return PostConfiguredTargetValue.key(
          new LabelAndConfiguration(input.getLabel(), input.getConfiguration()));
    }
  };

  private final SequencedSkyframeExecutor.BuildViewProvider buildViewProvider;

  public PostConfiguredTargetFunction(
      SequencedSkyframeExecutor.BuildViewProvider buildViewProvider) {
    this.buildViewProvider = Preconditions.checkNotNull(buildViewProvider);
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
    ImmutableMap<Action, Exception> badActions = BuildVariableValue.BAD_ACTIONS.get(env);
    ConfiguredTargetValue ctValue = (ConfiguredTargetValue)
        env.getValue(ConfiguredTargetValue.key((LabelAndConfiguration) skyKey.argument()));
    SkyframeDependencyResolver resolver =
        buildViewProvider.getSkyframeBuildView().createDependencyResolver(env);
    if (env.valuesMissing()) {
      return null;
    }

    for (Action action : ctValue.getActions()) {
      if (badActions.containsKey(action)) {
        throw new ActionConflictFunctionException(skyKey, badActions.get(action));
      }
    }

    ConfiguredTarget ct = ctValue.getConfiguredTarget();
    TargetAndConfiguration ctgValue =
        new TargetAndConfiguration(ct.getTarget(), ct.getConfiguration());

    env.getValues(Iterables.transform(resolver.dependentNodeMap(ctgValue).values(), TO_KEYS));
    if (env.valuesMissing()) {
      return null;
    }

    return new PostConfiguredTargetValue(ct);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((LabelAndConfiguration) skyKey.argument()).getLabel());
  }

  private static class ActionConflictFunctionException extends SkyFunctionException {
    public ActionConflictFunctionException(SkyKey skyKey, Throwable cause) {
      super(skyKey, cause);
    }
  }
}

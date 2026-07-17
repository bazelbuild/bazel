// Copyright 2020 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.auto.value.AutoValue;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.analysis.ConfiguredObjectValue;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.GraphTraversingHelper;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * Checks to see if any artifacts to be built by this {@link ActionLookupKey} transitively depend on
 * actions from an {@link ActionLookupValue} that has an action in conflict with another. If so,
 * none of this key's artifacts will be built.
 */
class TopLevelActionLookupConflictFindingFunction implements SkyFunction {
  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    Key key = (Key) skyKey;
    Pair<ConfiguredObjectValue, TopLevelArtifactHelper.ArtifactsToBuild> valueAndArtifactsToBuild =
        CompletionFunction.getValueAndArtifactsToBuild(key, env);
    if (env.valuesMissing()) {
      return null;
    }
    return GraphTraversingHelper.declareDependenciesAndCheckIfValuesMissingMaybeWithExceptions(
            env,
            ActionLookupConflictFindingFunction.convertArtifacts(
                    valueAndArtifactsToBuild.second.getAllArtifacts())
                .collect(toImmutableList()))
        ? null
        : ActionLookupConflictFindingValue.INSTANCE;
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((Key) skyKey).actionLookupKey().getLabel());
  }

  static Iterable<Key> keys(
      Iterable<ActionLookupKey> keys, TopLevelArtifactContext topLevelArtifactContext) {
    return Iterables.transform(keys, k -> Key.create(k, topLevelArtifactContext));
  }

  @AutoValue
  abstract static class Key implements TopLevelActionLookupKeyWrapper {
    static Key create(
        ActionLookupKey actionLookupKey, TopLevelArtifactContext topLevelArtifactContext) {
      return new AutoValue_TopLevelActionLookupConflictFindingFunction_Key(
          actionLookupKey, topLevelArtifactContext);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.TOP_LEVEL_ACTION_LOOKUP_CONFLICT_FINDING;
    }
  }
}

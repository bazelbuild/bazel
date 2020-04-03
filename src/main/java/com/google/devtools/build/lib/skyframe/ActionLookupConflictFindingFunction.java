// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skyframe.SkyframeActionExecutor.ConflictException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Set;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/** Check all transitive actions of an {@link ActionLookupValue} for action conflicts. */
public class ActionLookupConflictFindingFunction implements SkyFunction {
  ActionLookupConflictFindingFunction() {}

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    ImmutableMap<ActionAnalysisMetadata, ConflictException> badActions =
        PrecomputedValue.BAD_ACTIONS.get(env);
    ActionLookupValue alValue =
        (ActionLookupValue)
            env.getValue(((ActionLookupConflictFindingValue.Key) skyKey).argument());
    if (env.valuesMissing()) {
      return null;
    }

    Set<ActionLookupConflictFindingValue.Key> depKeys = CompactHashSet.create();
    for (ActionAnalysisMetadata action : alValue.getActions()) {
      if (badActions.containsKey(action)) {
        throw new ActionConflictFunctionException(badActions.get(action));
      }
      convertArtifacts(action.getInputs()).forEach(depKeys::add);
    }
    // Avoid silly cycles.
    depKeys.remove(skyKey);

    env.getValues(depKeys);
    return env.valuesMissing() ? null : ActionLookupConflictFindingValue.INSTANCE;
  }

  static Stream<ActionLookupConflictFindingValue.Key> convertArtifacts(
      NestedSet<Artifact> artifacts) {
    return artifacts.toList().stream()
        .filter(a -> !a.isSourceArtifact())
        .map(a -> (Artifact.DerivedArtifact) a)
        .map(
            a ->
                ActionLookupConflictFindingValue.key(
                    a.getGeneratingActionKey().getActionLookupKey()));
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((ConfiguredTargetKey) skyKey.argument()).getLabel());
  }

  private static class ActionConflictFunctionException extends SkyFunctionException {
    ActionConflictFunctionException(ConflictException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}

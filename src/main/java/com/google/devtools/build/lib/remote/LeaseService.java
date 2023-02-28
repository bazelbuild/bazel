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
package com.google.devtools.build.lib.remote;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCacheUtils;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import java.util.HashMap;
import java.util.Set;
import javax.annotation.Nullable;

/** A lease service that manages the lease of remote blobs. */
public class LeaseService {
  private final MemoizingEvaluator memoizingEvaluator;
  @Nullable private final ActionCache actionCache;

  public LeaseService(MemoizingEvaluator memoizingEvaluator, @Nullable ActionCache actionCache) {
    this.memoizingEvaluator = memoizingEvaluator;
    this.actionCache = actionCache;
  }

  /** Clean up internal state when files are evicted from remote CAS. */
  public void handleMissingInputs(Set<ActionInput> missingActionInputs) {
    if (missingActionInputs.isEmpty()) {
      return;
    }

    var actions = new HashMap<ActionLookupData, Action>();

    try {
      for (ActionInput actionInput : missingActionInputs) {
        if (actionInput instanceof Artifact.DerivedArtifact) {
          Artifact.DerivedArtifact output = (Artifact.DerivedArtifact) actionInput;
          ActionLookupData actionLookupData = output.getGeneratingActionKey();
          var actionLookupValue =
              memoizingEvaluator.getExistingValue(actionLookupData.getActionLookupKey());
          if (actionLookupValue instanceof ActionLookupValue) {
            Action action =
                ((ActionLookupValue) actionLookupValue)
                    .getAction(actionLookupData.getActionIndex());
            actions.put(actionLookupData, action);
          }
        }
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }

    if (!actions.isEmpty()) {
      var actionKeys = actions.keySet();
      memoizingEvaluator.delete(key -> key instanceof ActionLookupData && actionKeys.contains(key));

      if (actionCache != null) {
        for (var action : actions.values()) {
          ActionCacheUtils.removeCacheEntry(actionCache, action);
        }
      }
    }
  }
}

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

package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.actions.Artifact.OwnerlessArtifactWrapper;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import javax.annotation.Nullable;
import javax.annotation.concurrent.ThreadSafe;

/** An action graph that resolves generating actions by looking them up in a map. */
@ThreadSafe
public final class MapBasedActionGraph implements MutableActionGraph {

  private final ActionKeyContext actionKeyContext;
  private final ConcurrentMap<OwnerlessArtifactWrapper, ActionAnalysisMetadata> generatingActionMap;

  public MapBasedActionGraph(ActionKeyContext actionKeyContext) {
    this(actionKeyContext, /*sizeHint=*/ 16);
  }

  public MapBasedActionGraph(ActionKeyContext actionKeyContext, int sizeHint) {
    this.actionKeyContext = actionKeyContext;
    this.generatingActionMap = new ConcurrentHashMap<>(sizeHint);
  }

  @Override
  @Nullable
  public ActionAnalysisMetadata getGeneratingAction(Artifact artifact) {
    return generatingActionMap.get(new OwnerlessArtifactWrapper(artifact));
  }

  @Override
  public void registerAction(ActionAnalysisMetadata action)
      throws ActionConflictException, InterruptedException {
    for (Artifact artifact : action.getOutputs()) {
      ActionAnalysisMetadata previousAction =
          generatingActionMap.putIfAbsent(new OwnerlessArtifactWrapper(artifact), action);
      if (previousAction != null && previousAction != action) {
        if (Actions.canBeSharedLogForPotentialFalsePositives(
            actionKeyContext, action, previousAction)) {
          return; // All outputs can be shared. No need to register the remaining outputs.
        }
        throw new ActionConflictException(actionKeyContext, artifact, previousAction, action);
      }
    }
  }

  public int getSize() {
    return generatingActionMap.size();
  }
}

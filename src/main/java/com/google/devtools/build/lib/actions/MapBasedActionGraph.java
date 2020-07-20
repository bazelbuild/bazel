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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact.OwnerlessArtifactWrapper;
import javax.annotation.Nullable;
import javax.annotation.concurrent.ThreadSafe;

/**
 * An action graph that resolves generating actions by looking them up in a map.
 */
@ThreadSafe
public final class MapBasedActionGraph implements MutableActionGraph {
  private final ActionKeyContext actionKeyContext;
  private final ConcurrentMultimapWithHeadElement<OwnerlessArtifactWrapper, ActionAnalysisMetadata>
      generatingActionMap = new ConcurrentMultimapWithHeadElement<>();

  public MapBasedActionGraph(ActionKeyContext actionKeyContext) {
    this.actionKeyContext = actionKeyContext;
  }

  @Override
  @Nullable
  public ActionAnalysisMetadata getGeneratingAction(Artifact artifact) {
    return generatingActionMap.get(new OwnerlessArtifactWrapper(artifact));
  }

  @Override
  public void registerAction(ActionAnalysisMetadata action) throws ActionConflictException {
    for (Artifact artifact : action.getOutputs()) {
      OwnerlessArtifactWrapper wrapper = new OwnerlessArtifactWrapper(artifact);
      ActionAnalysisMetadata previousAction = generatingActionMap.putAndGet(wrapper, action);
      if (previousAction != null
          && previousAction != action
          && !Actions.canBeSharedLogForPotentialFalsePositives(
              actionKeyContext, action, previousAction)) {
        generatingActionMap.remove(wrapper, action);
        throw new ActionConflictException(actionKeyContext, artifact, previousAction, action);
      }
    }
  }

  @Override
  public void unregisterAction(ActionAnalysisMetadata action) {
    for (Artifact artifact : action.getOutputs()) {
      OwnerlessArtifactWrapper wrapper = new OwnerlessArtifactWrapper(artifact);
      generatingActionMap.remove(wrapper, action);
      ActionAnalysisMetadata otherAction = generatingActionMap.get(wrapper);
      Preconditions.checkState(
          otherAction == null
              || (otherAction != action
                  && Actions.canBeShared(actionKeyContext, action, otherAction)),
          "%s %s",
          action,
          otherAction);
    }
  }

  @Override
  public void clear() {
    generatingActionMap.clear();
  }
}

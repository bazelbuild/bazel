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

package com.google.devtools.build.lib.actions;

import com.google.common.base.Preconditions;

import javax.annotation.Nullable;
import javax.annotation.concurrent.ThreadSafe;

/**
 * An action graph that resolves generating actions by looking them up in a map.
 */
@ThreadSafe
public final class MapBasedActionGraph implements MutableActionGraph {
  private final ConcurrentMultimapWithHeadElement<Artifact, Action> generatingActionMap =
      new ConcurrentMultimapWithHeadElement<>();

  @Override
  @Nullable
  public Action getGeneratingAction(Artifact artifact) {
    return generatingActionMap.get(artifact);
  }

  @Override
  public void registerAction(Action action) throws ActionConflictException {
    for (Artifact artifact : action.getOutputs()) {
      Action previousAction = generatingActionMap.putAndGet(artifact, action);
      if (previousAction != null && previousAction != action
          && !Actions.canBeShared(action, previousAction)) {
        generatingActionMap.remove(artifact, action);
        throw new ActionConflictException(artifact, previousAction, action);
      }
    }
  }

  @Override
  public void unregisterAction(Action action) {
    for (Artifact artifact : action.getOutputs()) {
      generatingActionMap.remove(artifact, action);
      Action otherAction = generatingActionMap.get(artifact);
      Preconditions.checkState(otherAction == null
          || (otherAction != action && Actions.canBeShared(action, otherAction)),
          "%s %s", action, otherAction);
    }
  }

  @Override
  public void clear() {
    generatingActionMap.clear();
  }
}

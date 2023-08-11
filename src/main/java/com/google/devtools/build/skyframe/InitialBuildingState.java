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
package com.google.devtools.build.skyframe;

import com.google.devtools.build.skyframe.NodeEntry.DirtyType;
import javax.annotation.Nullable;

/**
 * {@link DirtyBuildingState} for a node on its initial build or a {@link
 * NonIncrementalInMemoryNodeEntry} being {@linkplain NodeEntry#forceRebuild force rebuilt}.
 */
class InitialBuildingState extends DirtyBuildingState {

  InitialBuildingState(boolean hasLowFanout) {
    super(DirtyType.CHANGE, hasLowFanout);
  }

  @Nullable
  @Override
  public final GroupedDeps getLastBuildDirectDeps() {
    return null;
  }

  @Override
  protected final int getNumOfGroupsInLastBuildDirectDeps() {
    return 0;
  }

  @Nullable
  @Override
  public final SkyValue getLastBuildValue() {
    return null;
  }

  @Override
  protected final boolean isIncremental() {
    return false;
  }
}

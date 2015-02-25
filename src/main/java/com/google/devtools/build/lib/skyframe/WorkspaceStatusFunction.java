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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/** Creates the workspace status artifacts and action. */
public class WorkspaceStatusFunction implements SkyFunction {
  WorkspaceStatusFunction() {
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) {
    Preconditions.checkState(
        WorkspaceStatusValue.SKY_KEY.equals(skyKey), WorkspaceStatusValue.SKY_KEY);

    WorkspaceStatusAction action = PrecomputedValue.WORKSPACE_STATUS_KEY.get(env);
    if (action == null) {
      return null;
    }

    return new WorkspaceStatusValue(
        action.getStableStatus(),
        action.getVolatileStatus(),
        action);
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

}

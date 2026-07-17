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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** Creates the workspace status artifacts and action. */
public class WorkspaceStatusFunction implements SkyFunction {

  private final Supplier<WorkspaceStatusAction> workspaceStatusActionFactory;

  WorkspaceStatusFunction(Supplier<WorkspaceStatusAction> workspaceStatusActionFactory) {
    this.workspaceStatusActionFactory = workspaceStatusActionFactory;
  }

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    Preconditions.checkState(
        WorkspaceStatusValue.BUILD_INFO_KEY.equals(skyKey), WorkspaceStatusValue.BUILD_INFO_KEY);
    WorkspaceStatusAction action = workspaceStatusActionFactory.get();

    ActionLookupData generatingActionKey =
        ActionLookupData.createUnshareable(WorkspaceStatusValue.BUILD_INFO_KEY, 0);
    for (Artifact output : action.getOutputs()) {
      ((DerivedArtifact) output).setGeneratingActionKey(generatingActionKey);
    }

    return new WorkspaceStatusValue(action);
  }
}

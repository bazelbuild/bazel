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
package com.google.devtools.build.lib.view;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableTable;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoCollection;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoFactory.BuildInfoKey;
import com.google.devtools.build.lib.view.config.BuildConfiguration;

import java.util.Collection;
import java.util.Map;

/** A container class for the workspace status artifacts and actions. */
public final class WorkspaceStatusArtifacts {
  private final ImmutableTable<String, BuildInfoKey, BuildInfoCollection> buildInfo;
  private final Artifact stableStatus;
  private final Artifact volatileStatus;
  private final ImmutableSet<Action> actions;
  private final WorkspaceStatusAction workspaceStatusAction;

  public WorkspaceStatusArtifacts(
      Artifact stableStatus, Artifact volatileStatus,
      ImmutableTable<String, BuildInfoKey, BuildInfoCollection> buildInfo,
      WorkspaceStatusAction buildInfoAction) {
    this.stableStatus = stableStatus;
    this.volatileStatus = volatileStatus;
    this.buildInfo = buildInfo;
    this.workspaceStatusAction = buildInfoAction;

    // The buildInfo table often contains multiple actions for the same output artifact.
    // Deduplicate the actions using a map.
    Map<Artifact, Action> actionsBuilder = Maps.newHashMap();
    for (BuildInfoCollection container : buildInfo.values()) {
      for (Action action : container.getActions()) {
        for (Artifact output : action.getOutputs()) {
          actionsBuilder.put(output, action);
        }
      }
    }
    for (Artifact output : buildInfoAction.getOutputs()) {
      actionsBuilder.put(output, buildInfoAction);
    }
    this.actions = ImmutableSet.copyOf(actionsBuilder.values());
  }

  Artifact getStableStatus() {
    return stableStatus;
  }

  Artifact getVolatileStatus() {
    return volatileStatus;
  }

  ImmutableList<Artifact> getBuildInfo(BuildConfiguration config, BuildInfoKey key, boolean stamp) {
    String configKey = config.cacheKey();
    BuildInfoCollection collection = buildInfo.get(configKey, key);
    if (collection == null) {
      return ImmutableList.of();
    }
    return stamp ? collection.getStampedBuildInfo() : collection.getRedactedBuildInfo();
  }

  public Collection<Action> getActions() {
    return actions;
  }

  WorkspaceStatusAction getBuildInfoAction() {
    return workspaceStatusAction;
  }
}

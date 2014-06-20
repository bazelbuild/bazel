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

import com.google.common.base.Preconditions;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableTable;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoCollection;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoFactory;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoFactory.BuildInfoContext;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoFactory.BuildInfoKey;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoFactory.BuildInfoType;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.config.BuildConfigurationCollection;

import java.util.UUID;

import javax.annotation.Nullable;

/**
 * Utility functions for performing the workspace status query at the start of
 * each build.
 */
@ThreadSafe
public final class WorkspaceStatusUtils {
  static WorkspaceStatusArtifacts createWorkspaceStatusArtifacts(Root buildInfoRoot,
      ImmutableList<BuildInfoFactory> buildInfoFactories, ArtifactFactory artifactFactory,
      BuildConfigurationCollection configurations, UUID buildId,
      WorkspaceStatusAction.Factory workspaceStatusActionFactory) {
    return new WorkspaceStatusUtils(buildInfoRoot, buildInfoFactories,
        artifactFactory, configurations, buildId, workspaceStatusActionFactory)
        .create();
  }

  private Root buildInfoRoot;
  private final ImmutableList<BuildInfoFactory> buildInfoFactories;
  private final ArtifactFactory artifactFactory;
  private final BuildConfigurationCollection configurations;
  private final WorkspaceStatusAction.Factory workspaceStatusActionFactory;
  @Nullable private final UUID buildId;

  private WorkspaceStatusUtils(Root buildInfoRoot,
      ImmutableList<BuildInfoFactory> buildInfoFactories, ArtifactFactory artifactFactory,
      BuildConfigurationCollection configurations,
      @Nullable UUID buildId, WorkspaceStatusAction.Factory workspaceStatusActionFactory) {
    this.buildInfoRoot = buildInfoRoot;
    this.buildInfoFactories = buildInfoFactories;
    this.artifactFactory = Preconditions.checkNotNull(artifactFactory);
    this.configurations = Preconditions.checkNotNull(configurations);
    this.workspaceStatusActionFactory = workspaceStatusActionFactory;
    this.buildId = buildId;
  }

  private WorkspaceStatusArtifacts create() {
    WorkspaceStatusAction workspaceStatusAction =
        workspaceStatusActionFactory.createWorkspaceStatusAction(
            artifactFactory, ArtifactOwner.NULL_OWNER, Suppliers.ofInstance(buildId));
    BuildInfoContext context = new BuildInfoContext() {
      @Override
      public Artifact getBuildInfoArtifact(
          PathFragment rootRelativePath, Root root, BuildInfoType type) {
        return artifactFactory.getSpecialMetadataHandlingArtifact(rootRelativePath, root,
            ArtifactOwner.NULL_OWNER, type == BuildInfoType.NO_REBUILD,
            type != BuildInfoType.NO_REBUILD);
      }

      @Override
      public Root getBuildDataDirectory() {
        return buildInfoRoot;
      }
    };
    Artifact stableStatus = workspaceStatusAction.getStableStatus();
    Artifact volatileStatus = workspaceStatusAction.getVolatileStatus();

    ImmutableTable.Builder<String, BuildInfoKey, BuildInfoCollection> buildInfoBuilder =
        ImmutableTable.builder();
    for (BuildInfoFactory factory : buildInfoFactories) {
      for (BuildConfiguration config : configurations.getAllConfigurations()) {
        BuildInfoCollection collection =
            factory.create(context, config, stableStatus, volatileStatus);
        buildInfoBuilder.put(config.cacheKey(), factory.getKey(), collection);
      }
    }
    return new WorkspaceStatusArtifacts(stableStatus, volatileStatus,
        buildInfoBuilder.build(), workspaceStatusAction);
  }
}

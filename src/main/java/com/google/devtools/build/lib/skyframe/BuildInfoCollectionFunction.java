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

import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory.BuildInfoContext;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory.BuildInfoKey;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory.BuildInfoType;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.BuildInfoCollectionValue.BuildInfoKeyAndConfig;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * Creates a {@link BuildInfoCollectionValue}. Only depends on the unique
 * {@link WorkspaceStatusValue} and the constant {@link PrecomputedValue#BUILD_INFO_FACTORIES}
 * injected value.
 */
public class BuildInfoCollectionFunction implements SkyFunction {
  // Supplier only because the artifact factory has not yet been created at constructor time.
  private final Supplier<ArtifactFactory> artifactFactory;
  private final Supplier<Boolean> removeActionsAfterEvaluation;
  private final ImmutableMap<BuildInfoKey, BuildInfoFactory> buildInfoFactories;

  BuildInfoCollectionFunction(
      Supplier<ArtifactFactory> artifactFactory,
      ImmutableMap<BuildInfoKey, BuildInfoFactory> buildInfoFactories,
      Supplier<Boolean> removeActionsAfterEvaluation) {
    this.artifactFactory = artifactFactory;
    this.buildInfoFactories = buildInfoFactories;
    this.removeActionsAfterEvaluation = Preconditions.checkNotNull(removeActionsAfterEvaluation);
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    final BuildInfoKeyAndConfig keyAndConfig = (BuildInfoKeyAndConfig) skyKey.argument();
    WorkspaceStatusValue infoArtifactValue =
        (WorkspaceStatusValue) env.getValue(WorkspaceStatusValue.SKY_KEY);
    if (infoArtifactValue == null) {
      return null;
    }
    WorkspaceNameValue nameValue = (WorkspaceNameValue) env.getValue(WorkspaceNameValue.key());
    if (nameValue == null) {
      return null;
    }
    RepositoryName repositoryName = RepositoryName.createFromValidStrippedName(
        nameValue.getName());

    final ArtifactFactory factory = artifactFactory.get();
    BuildInfoContext context = new BuildInfoContext() {
      @Override
      public Artifact getBuildInfoArtifact(PathFragment rootRelativePath, Root root,
          BuildInfoType type) {
        return type == BuildInfoType.NO_REBUILD
            ? factory.getConstantMetadataArtifact(rootRelativePath, root, keyAndConfig)
            : factory.getDerivedArtifact(rootRelativePath, root, keyAndConfig);
      }
    };

    return new BuildInfoCollectionValue(
        buildInfoFactories
            .get(keyAndConfig.getInfoKey())
            .create(
                context,
                keyAndConfig.getConfig(),
                infoArtifactValue.getStableArtifact(),
                infoArtifactValue.getVolatileArtifact(),
                repositoryName),
        removeActionsAfterEvaluation.get());
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}

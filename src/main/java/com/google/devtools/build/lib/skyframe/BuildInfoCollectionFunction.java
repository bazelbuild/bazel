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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Actions.GeneratingActions;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoCollection;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory.BuildInfoContext;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory.BuildInfoType;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoKey;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.skyframe.BuildInfoCollectionValue.BuildInfoKeyAndConfig;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Precomputed;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;

/**
 * Creates a {@link BuildInfoCollectionValue}. Only depends on the unique {@link
 * WorkspaceStatusValue} and the constant {@link #BUILD_INFO_FACTORIES} injected value.
 */
public class BuildInfoCollectionFunction implements SkyFunction {

  public static final Precomputed<Map<BuildInfoKey, BuildInfoFactory>> BUILD_INFO_FACTORIES =
      new Precomputed<>("build_info_factories");

  private final ActionKeyContext actionKeyContext;
  private final ArtifactFactory artifactFactory;

  BuildInfoCollectionFunction(ActionKeyContext actionKeyContext, ArtifactFactory artifactFactory) {
    this.actionKeyContext = actionKeyContext;
    this.artifactFactory = artifactFactory;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    final BuildInfoKeyAndConfig keyAndConfig = (BuildInfoKeyAndConfig) skyKey.argument();
    ImmutableSet<SkyKey> keysToRequest =
        ImmutableSet.of(
            WorkspaceStatusValue.BUILD_INFO_KEY,
            WorkspaceNameValue.key(),
            keyAndConfig.getConfigKey());
    Map<SkyKey, SkyValue> result = env.getValues(keysToRequest);
    if (env.valuesMissing()) {
      return null;
    }
    WorkspaceStatusValue infoArtifactValue =
        (WorkspaceStatusValue) result.get(WorkspaceStatusValue.BUILD_INFO_KEY);

    BuildConfiguration config =
        ((BuildConfigurationValue) result.get(keyAndConfig.getConfigKey())).getConfiguration();
    Map<BuildInfoKey, BuildInfoFactory> buildInfoFactories = BUILD_INFO_FACTORIES.get(env);
    BuildInfoFactory buildInfoFactory = buildInfoFactories.get(keyAndConfig.getInfoKey());
    Preconditions.checkState(buildInfoFactory.isEnabled(config));

    BuildInfoContext context =
        (rootRelativePath, root, type) ->
            type == BuildInfoType.NO_REBUILD
                ? artifactFactory.getConstantMetadataArtifact(rootRelativePath, root, keyAndConfig)
                : artifactFactory.getDerivedArtifact(rootRelativePath, root, keyAndConfig);
    BuildInfoCollection collection =
        buildInfoFactory.create(
            context,
            config,
            infoArtifactValue.getStableArtifact(),
            infoArtifactValue.getVolatileArtifact());
    GeneratingActions generatingActions;
    try {
      generatingActions =
          Actions.assignOwnersAndFilterSharedActionsAndThrowActionConflict(
              actionKeyContext,
              collection.getActions(),
              keyAndConfig,
              /*outputFiles=*/ null);
    } catch (ActionConflictException e) {
      throw new IllegalStateException("Action conflicts not expected in build info: " + skyKey, e);
    }
    return new BuildInfoCollectionValue(collection, generatingActions);
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}

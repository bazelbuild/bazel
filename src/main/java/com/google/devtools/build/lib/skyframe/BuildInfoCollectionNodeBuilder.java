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

import com.google.common.base.Supplier;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.skyframe.BuildInfoCollectionNode.BuildInfoKeyAndConfig;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoFactory;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoFactory.BuildInfoContext;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoFactory.BuildInfoKey;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoFactory.BuildInfoType;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeBuilder;
import com.google.devtools.build.skyframe.NodeKey;

import java.util.Map;

/**
 * Creates a {@link BuildInfoCollectionNode}. Only depends on the unique
 * {@link WorkspaceStatusNode} and the constant {@link BuildVariableNode#BUILD_INFO_FACTORIES}
 * injected node.
 */
public class BuildInfoCollectionNodeBuilder implements NodeBuilder {
  // Supplier only because the artifact factory has not yet been created at constructor time.
  private final Supplier<ArtifactFactory> artifactFactory;
  private final Root buildDataDirectory;

  BuildInfoCollectionNodeBuilder(Supplier<ArtifactFactory> artifactFactory,
      Root buildDataDirectory) {
    this.artifactFactory = artifactFactory;
    this.buildDataDirectory = buildDataDirectory;
  }

  @Override
  public Node build(NodeKey nodeKey, Environment env) {
    final BuildInfoKeyAndConfig keyAndConfig = (BuildInfoKeyAndConfig) nodeKey.getNodeName();
    WorkspaceStatusNode infoArtifactNode =
        (WorkspaceStatusNode) env.getDep(WorkspaceStatusNode.NODE_KEY);
    if (infoArtifactNode == null) {
      return null;
    }
    Map<BuildInfoKey, BuildInfoFactory> buildInfoFactories =
        BuildVariableNode.BUILD_INFO_FACTORIES.get(env);
    if (buildInfoFactories == null) {
      return null;
    }
    final ArtifactFactory factory = artifactFactory.get();
    BuildInfoContext context = new BuildInfoContext() {
      @Override
      public Artifact getBuildInfoArtifact(PathFragment rootRelativePath, Root root,
          BuildInfoType type) {
        return factory.getSpecialMetadataHandlingArtifact(rootRelativePath, root,
            keyAndConfig, type == BuildInfoType.NO_REBUILD, type != BuildInfoType.NO_REBUILD);
      }

      @Override
      public Root getBuildDataDirectory() {
        return buildDataDirectory;
      }
    };
    return new BuildInfoCollectionNode(buildInfoFactories.get(
        keyAndConfig.getInfoKey()).create(context, keyAndConfig.getConfig(),
            infoArtifactNode.getStableArtifact(), infoArtifactNode.getVolatileArtifact()));
  }

  @Override
  public String extractTag(NodeKey nodeKey) {
    return null;
  }
}

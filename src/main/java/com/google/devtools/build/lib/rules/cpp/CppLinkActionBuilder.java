// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.PathFragment;

/** Builder class to construct C++ link action. */
public class CppLinkActionBuilder {
  /**
   * Provides ActionConstructionContext, BuildConfigurationValue and methods for creating
   * intermediate and output artifacts for C++ linking.
   *
   * <p>This is unfortunately necessary, because most of the time, these artifacts are well-behaved
   * ones sitting under a package directory, but nativedeps link actions can be shared. In order to
   * avoid creating every artifact here with {@code getShareableArtifact()}, we abstract the
   * artifact creation away.
   *
   * <p>With shareableArtifacts set to true the implementation can create artifacts anywhere.
   *
   * <p>Necessary when the LTO backend actions of libraries should be shareable, and thus cannot be
   * under the package directory.
   *
   * <p>Necessary because the actions of nativedeps libraries should be shareable, and thus cannot
   * be under the package directory.
   */
  public static class LinkActionConstruction {
    private final boolean shareableArtifacts;
    private final ActionConstructionContext context;
    private final BuildConfigurationValue config;

    public ActionConstructionContext getContext() {
      return context;
    }

    public BuildConfigurationValue getConfig() {
      return config;
    }

    LinkActionConstruction(
        ActionConstructionContext context,
        BuildConfigurationValue config,
        boolean shareableArtifacts) {
      this.context = context;
      this.config = config;
      this.shareableArtifacts = shareableArtifacts;
    }

    public Artifact create(PathFragment rootRelativePath) {
      RepositoryName repositoryName = context.getActionOwner().getLabel().getRepository();
      if (shareableArtifacts) {
        return context.getShareableArtifact(
            rootRelativePath, config.getBinDirectory(repositoryName));

      } else {
        return context.getDerivedArtifact(rootRelativePath, config.getBinDirectory(repositoryName));
      }
    }

    public SpecialArtifact createTreeArtifact(PathFragment rootRelativePath) {
      RepositoryName repositoryName = context.getActionOwner().getLabel().getRepository();
      if (shareableArtifacts) {
        return context
            .getAnalysisEnvironment()
            .getTreeArtifact(rootRelativePath, config.getBinDirectory(repositoryName));
      } else {
        return context.getTreeArtifact(rootRelativePath, config.getBinDirectory(repositoryName));
      }
    }

    public ArtifactRoot getBinDirectory() {
      return config.getBinDirectory(context.getActionOwner().getLabel().getRepository());
    }
  }

  public static LinkActionConstruction newActionConstruction(RuleContext context) {
    return new LinkActionConstruction(context, context.getConfiguration(), false);
  }

  public static LinkActionConstruction newActionConstruction(
      ActionConstructionContext context,
      BuildConfigurationValue config,
      boolean shareableArtifacts) {
    return new LinkActionConstruction(context, config, shareableArtifacts);
  }

}

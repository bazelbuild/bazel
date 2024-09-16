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
package com.google.devtools.build.lib.actions;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/**
 * A factory to create middleman objects.
 */
@ThreadSafe
public final class MiddlemanFactory {

  private final ArtifactFactory artifactFactory;
  private final ActionRegistry actionRegistry;

  public MiddlemanFactory(
      ArtifactFactory artifactFactory, ActionRegistry actionRegistry) {
    this.artifactFactory = Preconditions.checkNotNull(artifactFactory);
    this.actionRegistry = Preconditions.checkNotNull(actionRegistry);
  }

  /**
   * Returns <code>null</code> iff inputs is empty. Returns the sole element of inputs iff <code>
   * inputs.size()==1</code>. Otherwise, returns a middleman artifact and creates a middleman action
   * that generates that artifact.
   *
   * @param owner the owner of the action that will be created.
   * @param owningArtifact the artifact of the file for which the runfiles should be created. There
   *     may be at most one set of runfiles for an owning artifact, unless the owning artifact is
   *     null. There may be at most one set of runfiles per owner with a null owning artifact.
   *     Further, if the owning Artifact is non-null, the owning Artifacts' root-relative path must
   *     be unique and the artifact must be part of the runfiles tree for which this middleman is
   *     created. Usually this artifact will be an executable program.
   * @param runfilesTree the runfiles tree for which the middleman being created for
   * @param runfilesManifest the runfiles manifest for the runfiles tree
   * @param repoMappingManifest the repo mapping manifest for the runfiles tree
   * @param middlemanDir the directory in which to place the middleman.
   */
  public Artifact createRunfilesMiddleman(
      ActionOwner owner,
      @Nullable Artifact owningArtifact,
      RunfilesTree runfilesTree,
      @Nullable Artifact runfilesManifest,
      @Nullable Artifact repoMappingManifest,
      ArtifactRoot middlemanDir) {
    Preconditions.checkArgument(middlemanDir.isMiddlemanRoot());

    NestedSetBuilder<Artifact> depsBuilder = NestedSetBuilder.stableOrder();
    depsBuilder.addTransitive(runfilesTree.getArtifacts());
    if (runfilesManifest != null) {
      depsBuilder.add(runfilesManifest);
    }
    if (repoMappingManifest != null) {
      depsBuilder.add(repoMappingManifest);
    }

    NestedSet<Artifact> deps = depsBuilder.build();
    String middlemanPath = owningArtifact == null
       ? Label.print(owner.getLabel())
       : owningArtifact.getRootRelativePath().getPathString();
    String escapedFilename = Actions.escapedPath(middlemanPath);
    PathFragment stampName = PathFragment.create("_middlemen/" + escapedFilename + "-runfiles");
    Artifact.DerivedArtifact stampFile =
        artifactFactory.getDerivedArtifact(stampName, middlemanDir, actionRegistry.getOwner());
    MiddlemanAction middlemanAction =
        new MiddlemanAction(owner, runfilesTree, deps, ImmutableSet.of(stampFile));
    actionRegistry.registerAction(middlemanAction);
    return stampFile;
  }

}

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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Pair;
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
   * @param inputs the set of artifacts for which the created artifact is to be the middleman.
   * @param middlemanDir the directory in which to place the middleman.
   */
  public Artifact createRunfilesMiddleman(
      ActionOwner owner,
      @Nullable Artifact owningArtifact,
      NestedSet<Artifact> inputs,
      ArtifactRoot middlemanDir,
      String tag) {
    Preconditions.checkArgument(middlemanDir.isMiddlemanRoot());
    if (inputs.isSingleton()) { // Optimization: No middleman for just one input.
      return inputs.getSingleton();
    }
    String middlemanPath = owningArtifact == null
       ? Label.print(owner.getLabel())
       : owningArtifact.getRootRelativePath().getPathString();
    return createMiddleman(owner, middlemanPath, tag, inputs, middlemanDir,
        MiddlemanType.RUNFILES_MIDDLEMAN).getFirst();
  }

  /**
   * Creates middlemen.
   *
   * <p>Note: there's no need to synchronize this method; the only use of a field is via a call to
   * another synchronized method (getArtifact()).
   *
   * @return null iff {@code inputs} is null or empty; the middleman file and the middleman action
   *     otherwise
   */
  @Nullable
  private Pair<Artifact, Action> createMiddleman(
      ActionOwner owner,
      String middlemanName,
      String purpose,
      NestedSet<Artifact> inputs,
      ArtifactRoot middlemanDir,
      MiddlemanType middlemanType) {
    if (inputs == null || inputs.isEmpty()) {
      return null;
    }

    Artifact stampFile = getStampFileArtifact(middlemanName, purpose, middlemanDir);
    Action action =
        MiddlemanAction.create(actionRegistry, owner, inputs, stampFile, purpose, middlemanType);
    return Pair.of(stampFile, action);
  }

  private Artifact.DerivedArtifact getStampFileArtifact(
      String middlemanName, String purpose, ArtifactRoot middlemanDir) {
    String escapedFilename = Actions.escapedPath(middlemanName);
    PathFragment stampName = PathFragment.create("_middlemen/" + escapedFilename + "-" + purpose);
    Artifact.DerivedArtifact stampFile =
        artifactFactory.getDerivedArtifact(stampName, middlemanDir, actionRegistry.getOwner());
    return stampFile;
  }
}

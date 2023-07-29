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
package com.google.devtools.build.lib.analysis.actions;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/**
 * Action responsible for the symlink tree creation.
 * Used to generate runfiles and fileset symlink farms.
 */
@Immutable
public final class SymlinkTreeAction extends AbstractAction {

  private static final String GUID = "7a16371c-cd4a-494d-b622-963cd89f5212";

  private final Artifact inputManifest;
  private final Runfiles runfiles;
  private final Artifact outputManifest;
  @Nullable private final String filesetRoot;
  private final ActionEnvironment env;
  private final boolean enableRunfiles;
  private final boolean inprocessSymlinkCreation;
  private final Artifact repoMappingManifest;

  /**
   * Creates SymlinkTreeAction instance.
   *
   * @param owner action owner
   * @param config the action owners build configuration
   * @param inputManifest the input runfiles manifest
   * @param runfiles the input runfiles
   * @param outputManifest the generated symlink tree manifest (must have "MANIFEST" base name).
   *     Symlink tree root will be set to the artifact's parent directory.
   * @param repoMappingManifest the repository mapping manifest
   * @param filesetRoot non-null if this is a fileset symlink tree
   */
  public SymlinkTreeAction(
      ActionOwner owner,
      BuildConfigurationValue config,
      Artifact inputManifest,
      @Nullable Runfiles runfiles,
      Artifact outputManifest,
      @Nullable Artifact repoMappingManifest,
      String filesetRoot) {
    this(
        owner,
        inputManifest,
        runfiles,
        outputManifest,
        repoMappingManifest,
        filesetRoot,
        config.getActionEnvironment(),
        config.runfilesEnabled(),
        config.inprocessSymlinkCreation());
  }

  /**
   * Creates SymlinkTreeAction instance. Prefer the constructor that takes a {@link
   * BuildConfigurationValue} instance; it is less likely to require changes in the future if we add
   * more command-line flags that affect this action.
   *
   * @param owner action owner
   * @param inputManifest the input runfiles manifest
   * @param runfiles the input runfiles
   * @param outputManifest the generated symlink tree manifest (must have "MANIFEST" base name).
   *     Symlink tree root will be set to the artifact's parent directory.
   * @param repoMappingManifest the repository mapping manifest
   * @param filesetRoot non-null if this is a fileset symlink tree,
   */
  @VisibleForTesting
  public SymlinkTreeAction(
      ActionOwner owner,
      Artifact inputManifest,
      @Nullable Runfiles runfiles,
      Artifact outputManifest,
      @Nullable Artifact repoMappingManifest,
      @Nullable String filesetRoot,
      ActionEnvironment env,
      boolean enableRunfiles,
      boolean inprocessSymlinkCreation) {
    super(
        owner,
        computeInputs(enableRunfiles, runfiles, inputManifest, repoMappingManifest),
        ImmutableSet.of(outputManifest));
    Preconditions.checkArgument(outputManifest.getPath().getBaseName().equals("MANIFEST"));
    Preconditions.checkArgument(
        (runfiles == null) == (filesetRoot != null),
        "Runfiles must be null iff this is a fileset action");
    this.runfiles = runfiles;
    this.outputManifest = outputManifest;
    this.filesetRoot = filesetRoot;
    this.env = env;
    this.enableRunfiles = enableRunfiles;
    this.inprocessSymlinkCreation = inprocessSymlinkCreation;
    this.inputManifest = inputManifest;
    this.repoMappingManifest = repoMappingManifest;
  }

  private static NestedSet<Artifact> computeInputs(
      boolean enableRunfiles,
      Runfiles runfiles,
      Artifact inputManifest,
      @Nullable Artifact repoMappingManifest) {
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.stableOrder();
    inputs.add(inputManifest);
    // All current strategies (in-process and build-runfiles-windows) for
    // making symlink trees on Windows depend on the target files
    // existing, so directory or file links can be made as appropriate.
    if (enableRunfiles && runfiles != null && OS.getCurrent() == OS.WINDOWS) {
      inputs.addTransitive(runfiles.getAllArtifacts());
      if (repoMappingManifest != null) {
        inputs.add(repoMappingManifest);
      }
    }
    return inputs.build();
  }

  @Override
  public ActionEnvironment getEnvironment() {
    return env;
  }

  public Artifact getInputManifest() {
    return inputManifest;
  }

  @Nullable
  public Runfiles getRunfiles() {
    return runfiles;
  }

  public Artifact getOutputManifest() {
    return outputManifest;
  }

  @Nullable
  public Artifact getRepoMappingManifest() {
    return repoMappingManifest;
  }

  public boolean isFilesetTree() {
    return filesetRoot != null;
  }

  public PathFragment getFilesetRoot() {
    return PathFragment.create(filesetRoot);
  }

  public boolean isRunfilesEnabled() {
    return enableRunfiles;
  }

  public boolean inprocessSymlinkCreation() {
    return inprocessSymlinkCreation;
  }

  @Override
  public String getMnemonic() {
    return "SymlinkTree";
  }

  @Override
  protected String getRawProgressMessage() {
    return (isFilesetTree() ? "Creating Fileset tree " : "Creating runfiles tree ")
        + outputManifest.getExecPath().getParentDirectory().getPathString();
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable Artifact.ArtifactExpander artifactExpander,
      Fingerprint fp) {
    fp.addString(GUID);
    fp.addNullableString(filesetRoot);
    fp.addBoolean(enableRunfiles);
    fp.addBoolean(inprocessSymlinkCreation);
    env.addTo(fp);
    // We need to ensure that the fingerprints for two different instances of this action are
    // different. Consider the hypothetical scenario where we add a second runfiles object to this
    // class, which could also be null: the sequence
    //    if (r1 != null) r1.fingerprint(fp);
    //    if (r2 != null) r2.fingerprint(fp);
    // would *not* be safe; we'd get a collision between an action that has only r1 set, and another
    // that has only r2 set. Prefixing with a boolean indicating the presence of runfiles makes it
    // safe to add more fields in the future.
    fp.addBoolean(runfiles != null);
    if (runfiles != null) {
      runfiles.fingerprint(actionKeyContext, fp);
    }
    fp.addBoolean(repoMappingManifest != null);
    if (repoMappingManifest != null) {
      fp.addPath(repoMappingManifest.getExecPath());
    }
  }

  @Override
  public ActionResult execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    actionExecutionContext
        .getContext(SymlinkTreeActionContext.class)
        .createSymlinks(this, actionExecutionContext);
    return ActionResult.EMPTY;
  }

  @Override
  public boolean mayInsensitivelyPropagateInputs() {
    return true;
  }
}

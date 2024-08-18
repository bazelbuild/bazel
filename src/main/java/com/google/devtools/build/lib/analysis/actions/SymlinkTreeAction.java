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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactExpander;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue.RunfileSymlinksMode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.OS;
import javax.annotation.Nullable;

/**
 * Action responsible for the symlink tree creation. Used to generate runfiles and fileset symlink
 * trees.
 */
@Immutable
public final class SymlinkTreeAction extends AbstractAction {

  private static final String GUID = "7a16371c-cd4a-494d-b622-963cd89f5212";

  private final Artifact inputManifest;
  private final Artifact outputManifest;
  private final ActionEnvironment env;
  private final RunfileSymlinksMode runfileSymlinksMode;
  private final Artifact repoMappingManifest;

  // Exactly one of these two fields is non-null.
  @Nullable private final Runfiles runfiles;
  @Nullable private final String workspaceNameForFileset;

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
   */
  public SymlinkTreeAction(
      ActionOwner owner,
      BuildConfigurationValue config,
      Artifact inputManifest,
      @Nullable Runfiles runfiles,
      Artifact outputManifest,
      @Nullable Artifact repoMappingManifest) {
    this(
        owner,
        inputManifest,
        runfiles,
        outputManifest,
        repoMappingManifest,
        config.getActionEnvironment(),
        config.getRunfileSymlinksMode(),
        config.getWorkspaceName());
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
   * @param workspaceName name of the workspace
   */
  @VisibleForTesting
  public SymlinkTreeAction(
      ActionOwner owner,
      Artifact inputManifest,
      @Nullable Runfiles runfiles,
      Artifact outputManifest,
      @Nullable Artifact repoMappingManifest,
      ActionEnvironment env,
      RunfileSymlinksMode runfileSymlinksMode,
      String workspaceName) {
    super(
        owner,
        computeInputs(runfileSymlinksMode, runfiles, inputManifest, repoMappingManifest),
        ImmutableSet.of(outputManifest));
    checkArgument(outputManifest.getExecPath().getBaseName().equals("MANIFEST"), outputManifest);
    this.outputManifest = outputManifest;
    this.env = env;
    this.runfileSymlinksMode = runfileSymlinksMode;
    this.inputManifest = inputManifest;
    this.repoMappingManifest = repoMappingManifest;
    if (inputManifest.isFileset()) {
      checkArgument(runfiles == null, "Runfiles present for fileset %s", inputManifest);
      this.runfiles = null;
      this.workspaceNameForFileset = checkNotNull(workspaceName);
    } else {
      this.runfiles = checkNotNull(runfiles);
      this.workspaceNameForFileset = null;
    }
  }

  private static NestedSet<Artifact> computeInputs(
      RunfileSymlinksMode runfileSymlinksMode,
      Runfiles runfiles,
      Artifact inputManifest,
      @Nullable Artifact repoMappingManifest) {
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.stableOrder();
    inputs.add(inputManifest);
    // All current strategies (in-process and build-runfiles-windows) for
    // making symlink trees on Windows depend on the target files
    // existing, so directory or file links can be made as appropriate.
    if (runfileSymlinksMode != RunfileSymlinksMode.SKIP
        && runfiles != null
        && OS.getCurrent() == OS.WINDOWS) {
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
    return workspaceNameForFileset != null;
  }

  public String getWorkspaceNameForFileset() {
    return checkNotNull(workspaceNameForFileset, "Not a fileset tree: %s", outputManifest);
  }

  public RunfileSymlinksMode getRunfileSymlinksMode() {
    return runfileSymlinksMode;
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
      @Nullable ArtifactExpander artifactExpander,
      Fingerprint fp) {
    fp.addString(GUID);
    fp.addNullableString(workspaceNameForFileset);
    fp.addInt(runfileSymlinksMode.ordinal());
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
      runfiles.fingerprint(actionKeyContext, fp, /* digestAbsolutePaths= */ true);
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

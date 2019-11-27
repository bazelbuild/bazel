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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.Fingerprint;
import javax.annotation.Nullable;

/**
 * Action responsible for the symlink tree creation.
 * Used to generate runfiles and fileset symlink farms.
 */
@Immutable
@AutoCodec
public final class SymlinkTreeAction extends AbstractAction {

  private static final String GUID = "63412bda-4026-4c8e-a3ad-7deb397728d4";

  private final Artifact inputManifest;
  private final Runfiles runfiles;
  private final Artifact outputManifest;
  private final boolean filesetTree;
  private final boolean enableRunfiles;

  /**
   * Creates SymlinkTreeAction instance.
   *
   * @param owner action owner
   * @param config the action owners build configuration
   * @param inputManifest the input runfiles manifest
   * @param runfiles the input runfiles
   * @param outputManifest the generated symlink tree manifest (must have "MANIFEST" base name).
   *     Symlink tree root will be set to the artifact's parent directory.
   * @param filesetTree true if this is fileset symlink tree
   */
  public SymlinkTreeAction(
      ActionOwner owner,
      BuildConfiguration config,
      Artifact inputManifest,
      @Nullable Runfiles runfiles,
      Artifact outputManifest,
      boolean filesetTree) {
    this(
        owner,
        inputManifest,
        runfiles,
        outputManifest,
        filesetTree,
        config.getActionEnvironment(),
        config.runfilesEnabled());
  }

  /**
   * Creates SymlinkTreeAction instance. Prefer the constructor that takes a {@link
   * BuildConfiguration} instance; it is less likely to require changes in the future if we add more
   * command-line flags that affect this action.
   *
   * @param owner action owner
   * @param inputManifest the input runfiles manifest
   * @param runfiles the input runfiles
   * @param outputManifest the generated symlink tree manifest (must have "MANIFEST" base name).
   *     Symlink tree root will be set to the artifact's parent directory.
   * @param filesetTree true if this is fileset symlink tree,
   */
  @AutoCodec.Instantiator
  public SymlinkTreeAction(
      ActionOwner owner,
      Artifact inputManifest,
      @Nullable Runfiles runfiles,
      Artifact outputManifest,
      boolean filesetTree,
      ActionEnvironment env,
      boolean enableRunfiles) {
    super(owner, ImmutableList.of(inputManifest), ImmutableList.of(outputManifest), env);
    Preconditions.checkArgument(outputManifest.getPath().getBaseName().equals("MANIFEST"));
    Preconditions.checkArgument(
        (runfiles == null) == filesetTree, "Runfiles must be null iff this is a fileset action");
    this.inputManifest = inputManifest;
    this.runfiles = runfiles;
    this.outputManifest = outputManifest;
    this.filesetTree = filesetTree;
    this.enableRunfiles = enableRunfiles;
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

  public boolean isFilesetTree() {
    return filesetTree;
  }

  public boolean isRunfilesEnabled() {
    return enableRunfiles;
  }

  @Override
  public String getMnemonic() {
    return "SymlinkTree";
  }

  @Override
  protected String getRawProgressMessage() {
    return (filesetTree ? "Creating Fileset tree " : "Creating runfiles tree ")
        + outputManifest.getExecPath().getParentDirectory().getPathString();
  }

  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {
    fp.addString(GUID);
    fp.addBoolean(filesetTree);
    fp.addBoolean(enableRunfiles);
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
      runfiles.fingerprint(fp);
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

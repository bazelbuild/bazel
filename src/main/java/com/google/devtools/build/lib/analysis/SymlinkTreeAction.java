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
package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Preconditions;

import javax.annotation.Nullable;

/**
 * Action responsible for the symlink tree creation.
 * Used to generate runfiles and fileset symlink farms.
 */
public class SymlinkTreeAction extends AbstractAction {

  private static final String GUID = "63412bda-4026-4c8e-a3ad-7deb397728d4";

  private final Artifact inputManifest;
  private final Artifact outputManifest;
  private final boolean filesetTree;

  /**
   * Creates SymlinkTreeAction instance.
   *
   * @param owner action owner
   * @param inputManifest the input runfiles manifest
   * @param artifactMiddleman the middleman artifact representing all the files the symlinks
   *                          point to (on Windows we need to know if the target of a "symlink" is
   *                          a directory or a file so we need to build it before)
   * @param outputManifest the generated symlink tree manifest
   *                       (must have "MANIFEST" base name). Symlink tree root
   *                       will be set to the artifact's parent directory.
   * @param filesetTree true if this is fileset symlink tree,
   *                    false if this is a runfiles symlink tree.
   */
  public SymlinkTreeAction(ActionOwner owner, Artifact inputManifest,
      @Nullable Artifact artifactMiddleman, Artifact outputManifest, boolean filesetTree) {
    super(owner, computeInputs(inputManifest, artifactMiddleman), ImmutableList.of(outputManifest));
    Preconditions.checkArgument(outputManifest.getPath().getBaseName().equals("MANIFEST"));
    this.inputManifest = inputManifest;
    this.outputManifest = outputManifest;
    this.filesetTree = filesetTree;
  }

  private static ImmutableList<Artifact> computeInputs(
      Artifact inputManifest, Artifact artifactMiddleman) {
    ImmutableList.Builder<Artifact> result = ImmutableList.<Artifact>builder()
        .add(inputManifest);
    if (artifactMiddleman != null
        && !artifactMiddleman.getPath().getFileSystem().supportsSymbolicLinksNatively()) {
      result.add(artifactMiddleman);
    }
    return result.build();
  }

  public Artifact getInputManifest() {
    return inputManifest;
  }

  public Artifact getOutputManifest() {
    return outputManifest;
  }

  public boolean isFilesetTree() {
    return filesetTree;
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
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addInt(filesetTree ? 1 : 0);
    return f.hexDigestAndReset();
  }

  @Override
  public ResourceSet estimateResourceConsumption(Executor executor) {
    // Return null here to indicate that resources would be managed manually
    // during action execution.
    return null;
  }

  @Override
  public void execute(
      ActionExecutionContext actionExecutionContext)
          throws ActionExecutionException, InterruptedException {
    actionExecutionContext.getExecutor().getContext(SymlinkTreeActionContext.class)
        .createSymlinks(this, actionExecutionContext);
  }
}

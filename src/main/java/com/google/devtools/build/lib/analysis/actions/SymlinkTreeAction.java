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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Action responsible for the symlink tree creation.
 * Used to generate runfiles and fileset symlink farms.
 */
@Immutable
@AutoCodec
public final class SymlinkTreeAction extends AbstractAction {

  private static final String GUID = "63412bda-4026-4c8e-a3ad-7deb397728d4";

  private final Artifact inputManifest;
  private final Artifact outputManifest;
  private final boolean filesetTree;
  private final boolean enableRunfiles;

  /**
   * Creates SymlinkTreeAction instance.
   *  @param owner action owner
   * @param inputManifest the input runfiles manifest
   * @param outputManifest the generated symlink tree manifest
   *                       (must have "MANIFEST" base name). Symlink tree root
   *                       will be set to the artifact's parent directory.
   * @param filesetTree true if this is fileset symlink tree,
   * @param enableRunfiles true is the actual symlink tree needs to be created.
   */
  @AutoCodec.Instantiator
  public SymlinkTreeAction(
      ActionOwner owner,
      Artifact inputManifest,
      Artifact outputManifest,
      boolean filesetTree,
      ActionEnvironment env,
      boolean enableRunfiles) {
    super(owner, ImmutableList.of(inputManifest), ImmutableList.of(outputManifest), env);
    Preconditions.checkArgument(outputManifest.getPath().getBaseName().equals("MANIFEST"));
    this.inputManifest = inputManifest;
    this.outputManifest = outputManifest;
    this.filesetTree = filesetTree;
    this.enableRunfiles = enableRunfiles;
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
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {
    fp.addString(GUID);
    fp.addBoolean(filesetTree);
    fp.addBoolean(enableRunfiles);
    env.addTo(fp);
  }

  @Override
  public ActionResult execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    Map<String, String> resolvedEnv = new LinkedHashMap<>();
    env.resolve(resolvedEnv, actionExecutionContext.getClientEnv());
    actionExecutionContext
        .getContext(SymlinkTreeActionContext.class)
        .createSymlinks(
            this, actionExecutionContext, ImmutableMap.copyOf(resolvedEnv), enableRunfiles);
    return ActionResult.EMPTY;
  }

  @Override
  public boolean mayInsensitivelyPropagateInputs() {
    return true;
  }
}

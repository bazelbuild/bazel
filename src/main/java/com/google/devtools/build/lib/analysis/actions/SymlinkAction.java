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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import javax.annotation.Nullable;

/** Action to create a symbolic link. */
@AutoCodec
public class SymlinkAction extends AbstractAction {
  private static final String GUID = "349675b5-437c-4da8-891a-7fb98fba6ab5";

  /** Null when {@link #getPrimaryInput} is the target of the symlink. */
  @Nullable private final PathFragment inputPath;

  private final String progressMessage;

  /**
   * Creates a new SymlinkAction instance.
   *
   * @param owner the action owner.
   * @param input the Artifact that will be the src of the symbolic link.
   * @param output the Artifact that will be created by executing this Action.
   * @param progressMessage the progress message.
   */
  public SymlinkAction(ActionOwner owner, Artifact input, Artifact output,
      String progressMessage) {
    // These actions typically have only one input and one output, which
    // become the sole and primary in their respective lists.
    super(owner, ImmutableList.of(input), ImmutableList.of(output));
    this.inputPath = null;
    this.progressMessage = progressMessage;
  }

  /**
   * Creates a new SymlinkAction instance, where the inputPath may be different than that input
   * artifact's path. This is only useful when dealing with runfiles trees where link target is a
   * directory.
   *
   * NB: Only use in special cases where the target of the symlink differs from the input
   * {@link Artifact}.
   *
   * @param owner the action owner.
   * @param inputPath the Path that will be the src of the symbolic link.
   * @param primaryInput the Artifact that is required to build the inputPath.
   * @param primaryOutput the Artifact that will be created by executing this Action.
   * @param progressMessage the progress message.
   */
  @AutoCodec.Instantiator
  public SymlinkAction(
      ActionOwner owner,
      PathFragment inputPath,
      Artifact primaryInput,
      Artifact primaryOutput,
      String progressMessage) {
    super(
        owner,
        primaryInput != null ? ImmutableList.of(primaryInput) : Artifact.NO_ARTIFACTS,
        ImmutableList.of(primaryOutput));
    this.inputPath = inputPath;
    this.progressMessage = progressMessage;
  }

  /**
   * Creates a new SymlinkAction instance, where an input artifact is not present. This is useful
   * when dealing with special cases where input paths that are outside the exec root directory
   * tree. Currently, the only instance where this happens is for FDO builds where the profile file
   * is outside the exec root structure.
   *
   * @param owner the action owner.
   * @param inputPath the Path that will be the src of the symbolic link.
   * @param output the Artifact that will be created by executing this Action.
   * @param progressMessage the progress message.
   */
  public SymlinkAction(
      ActionOwner owner, PathFragment inputPath, Artifact output, String progressMessage) {
    super(owner, Artifact.NO_ARTIFACTS, ImmutableList.of(output));
    // Don't use this constructor except when the symlink points to an absolute path
    Preconditions.checkState(inputPath.isAbsolute());
    this.inputPath = Preconditions.checkNotNull(inputPath);
    this.progressMessage = progressMessage;
  }

  public PathFragment getInputPath() {
    return inputPath == null ? getPrimaryInput().getExecPath() : inputPath;
  }

  public Path getOutputPath(ActionExecutionContext actionExecutionContext) {
    return actionExecutionContext.getInputPath(getPrimaryOutput());
  }

  @Override
  public ActionResult execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException {
    Path srcPath;
    if (inputPath == null) {
      srcPath = actionExecutionContext.getInputPath(getPrimaryInput());
    } else {
      srcPath = actionExecutionContext.getExecRoot().getRelative(inputPath);
    }
    try {
      getOutputPath(actionExecutionContext).createSymbolicLink(srcPath);
    } catch (IOException e) {
      throw new ActionExecutionException("failed to create symbolic link '"
          + Iterables.getOnlyElement(getOutputs()).prettyPrint()
          + "' to '" + printInputs()
          + "' due to I/O error: " + e.getMessage(), e, this, false);
    }
    return ActionResult.EMPTY;
  }

  private String printInputs() {
    if (Iterables.isEmpty(getInputs())) {
      return inputPath.getPathString();
    } else if (Iterables.size(getInputs()) == 1){
      return Iterables.getOnlyElement(getInputs()).prettyPrint();
    } else {
      throw new IllegalStateException(
          "Inputs unexpectedly contains more than 1 element: " + getInputs());
    }
  }

  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {
    fp.addString(GUID);
    // We don't normally need to add inputs to the key. In this case, however, the inputPath can be
    // different from the actual input artifact.
    if (inputPath != null) {
      fp.addPath(inputPath);
    }
  }

  @Override
  public String getMnemonic() {
    return "Symlink";
  }

  @Override
  public boolean isVolatile() {
    return inputPath != null && inputPath.isAbsolute();
  }

  @Override
  public boolean executeUnconditionally() {
    // If the SymlinkAction points to an absolute path, we can't verify that its output artifact did
    // not change purely by looking at the output tree. Thus, we re-execute the action just to be
    // safe. Change pruning will take care of not re-running dependent actions and this is used only
    // in very rare cases (only C++ FDO and even then, only twice per build at most) anyway.
    return inputPath != null && inputPath.isAbsolute();
  }

  @Override
  protected String getRawProgressMessage() {
    return progressMessage;
  }
}

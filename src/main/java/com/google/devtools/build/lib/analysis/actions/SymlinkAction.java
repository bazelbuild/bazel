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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * Action to create a symbolic link.
 */
public class SymlinkAction extends AbstractAction {

  private static final String GUID = "349675b5-437c-4da8-891a-7fb98fba6ab5";

  /** Null when {@link #getPrimaryInput} is the target of the symlink. */
  @Nullable private final PathFragment inputPath;

  private final Artifact output;
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
    this.output = Preconditions.checkNotNull(output);
    this.progressMessage = progressMessage;
  }

  /**
   * Creates a new SymlinkAction instance, where the inputPath
   * may be different than that input artifact's path. This is
   * only useful when dealing with runfiles trees where
   * link target is a directory.
   *
   * @param owner the action owner.
   * @param inputPath the Path that will be the src of the symbolic link.
   * @param input the Artifact that is required to build the inputPath.
   * @param output the Artifact that will be created by executing this Action.
   * @param progressMessage the progress message.
   */
  public SymlinkAction(ActionOwner owner, PathFragment inputPath, Artifact input,
      Artifact output, String progressMessage) {
    super(owner, ImmutableList.of(input), ImmutableList.of(output));
    this.inputPath = Preconditions.checkNotNull(inputPath);
    this.output = Preconditions.checkNotNull(output);
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
    this.inputPath = Preconditions.checkNotNull(inputPath);
    this.output = Preconditions.checkNotNull(output);
    this.progressMessage = progressMessage;
  }

  public PathFragment getInputPath() {
    return inputPath == null ? getPrimaryInput().getExecPath() : inputPath;
  }

  public Path getOutputPath() {
    return output.getPath();
  }

  @Override
  public void execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException {
    Path srcPath;
    if (inputPath == null) {
      srcPath = getPrimaryInput().getPath();
    } else {
      srcPath = actionExecutionContext.getExecRoot().getRelative(inputPath);
    }
    try {
      getOutputPath().createSymbolicLink(srcPath);
    } catch (IOException e) {
      throw new ActionExecutionException("failed to create symbolic link '"
          + Iterables.getOnlyElement(getOutputs()).prettyPrint()
          + "' to the '" + Iterables.getOnlyElement(getInputs()).prettyPrint()
          + "' due to I/O error: " + e.getMessage(), e, this, false);
    }
  }

  @Override
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    // We don't normally need to add inputs to the key. In this case, however, the inputPath can be
    // different from the actual input artifact.
    if (inputPath != null) {
      f.addPath(inputPath);
    }
    return f.hexDigestAndReset();
  }

  @Override
  public String getMnemonic() {
    return "Symlink";
  }

  @Override
  protected String getRawProgressMessage() {
    return progressMessage;
  }
}

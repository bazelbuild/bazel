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
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import javax.annotation.Nullable;

/** Action to create a symbolic link. */
@AutoCodec
public final class SymlinkAction extends AbstractAction {
  private static final String GUID = "7f4fab4d-d0a7-4f0f-8649-1d0337a21fee";

  /** Null when {@link #getPrimaryInput} is the target of the symlink. */
  @Nullable private final PathFragment inputPath;
  @Nullable private final String progressMessage;

  @VisibleForSerialization
  enum TargetType {
    /**
     * The symlink points into a Fileset.
     *
     * <p>If this is set, the action also updates the mtime for its target thus forcing actions
     * depending on it to be re-executed. This would not be necessary in an ideal world, but
     * dependency checking for Filesets output trees is unsound because they are directories, so we
     * need to force them to be considered changed this way. Yet Another Reason why Filests should
     * go away.
     */
    FILESET,

    /**
     * The symlink should point to an executable.
     *
     * <p>Blaze will verify that the target is indeed executable.
     */
    EXECUTABLE,

    /** Just a vanilla symlink. Don't do anything else other than creating the symlink. */
    OTHER,
  }

  private final TargetType targetType;

  /**
   * Creates an action that creates a symlink pointing to an artifact.
   *
   * @param owner the action owner.
   * @param input the {@link Artifact} the symlink will point to
   * @param output the {@link Artifact} that will be created by executing this Action.
   * @param progressMessage the progress message.
   */
  public static SymlinkAction toArtifact(ActionOwner owner, Artifact input, Artifact output,
      String progressMessage) {
    return new SymlinkAction(owner, null, input, output, progressMessage, TargetType.OTHER);
  }

  public static SymlinkAction toExecutable(
      ActionOwner owner, Artifact input, Artifact output, String progressMessage) {
    return new SymlinkAction(owner, null, input, output, progressMessage, TargetType.EXECUTABLE);
  }

  @VisibleForSerialization
  @AutoCodec.Instantiator
  protected SymlinkAction(
      ActionOwner owner,
      PathFragment inputPath,
      Artifact primaryInput,
      Artifact primaryOutput,
      String progressMessage,
      TargetType targetType) {
    super(
        owner,
        primaryInput != null ? ImmutableList.of(primaryInput) : Artifact.NO_ARTIFACTS,
        ImmutableList.of(primaryOutput));
    this.inputPath = inputPath;
    this.progressMessage = progressMessage;
    this.targetType = targetType;
  }

  /**
   * Creates a symlink to a Fileset.
   *
   * <p>This is different from a regular {@link SymlinkAction} in that the target is in the output
   * tree but not an artifact and that when running this action, the mtime of its target is updated
   * (necessary because dependency checking of Filesets is unsound). For more information, see the
   * Javadoc of {@code TargetType.FILESET}.
   *
   * <p><b>WARNING:</b>Do not use this for anything else other than Filesets. If you do, your
   * correctness will depend on a subtle interaction between various parts of Blaze.
   *
   * @param owner the action owner.
   * @param execPath where the symlink will point to
   * @param primaryInput the {@link Artifact} that is required to build the inputPath.
   * @param primaryOutput the {@link Artifact} that will be created by executing this Action.
   * @param progressMessage the progress message.
   */
  public static SymlinkAction toFileset(
      ActionOwner owner,
      PathFragment execPath,
      Artifact primaryInput,
      Artifact primaryOutput,
      String progressMessage) {
    Preconditions.checkState(!execPath.isAbsolute());
    return new SymlinkAction(
        owner, execPath, primaryInput, primaryOutput, progressMessage, TargetType.FILESET);
  }

  /**
   * Creates a new SymlinkAction instance, where an input artifact is not present. This is useful
   * when dealing with special cases where input paths that are outside the exec root directory
   * tree. Currently, the only instance where this happens is for FDO builds where the profile file
   * is outside the exec root structure.
   *
   * <p>Do <b>NOT</b> use this method unless there is no other way; unconditionally executed actions
   * are costly: even if change pruning kicks in and downstream actions are not re-executed, they
   * trigger unconditional Skyframe invalidation of their reverse dependencies.
   *
   * @param owner the action owner.
   * @param absolutePath where the symlink will point to
   * @param output the Artifact that will be created by executing this Action.
   * @param progressMessage the progress message.
   */
  public static SymlinkAction toAbsolutePath(ActionOwner owner, PathFragment absolutePath,
      Artifact output, String progressMessage) {
    Preconditions.checkState(absolutePath.isAbsolute());
    return new SymlinkAction(owner, absolutePath, null, output, progressMessage, TargetType.OTHER);
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
    maybeVerifyTargetIsExecutable(actionExecutionContext);

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

    updateInputMtimeIfNeeded(actionExecutionContext);
    return ActionResult.EMPTY;
  }

  private void maybeVerifyTargetIsExecutable(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException {
    if (targetType != TargetType.EXECUTABLE) {
      return;
    }

    Path inputPath = actionExecutionContext.getInputPath(getPrimaryInput());
    try {
      // Validate that input path is a file with the executable bit is set.
      if (!inputPath.isFile()) {
        throw new ActionExecutionException(
            "'" + Iterables.getOnlyElement(getInputs()).prettyPrint() + "' is not a file",
            this,
            false);
      }
      if (!inputPath.isExecutable()) {
        throw new ActionExecutionException(
            "failed to create symbolic link '"
                + Iterables.getOnlyElement(getOutputs()).prettyPrint()
                + "': file '"
                + Iterables.getOnlyElement(getInputs()).prettyPrint()
                + "' is not executable",
            this,
            false);
      }
    } catch (IOException e) {
      throw new ActionExecutionException(
          "failed to create symbolic link '"
              + Iterables.getOnlyElement(getOutputs()).prettyPrint()
              + "' to the '"
              + Iterables.getOnlyElement(getInputs()).prettyPrint()
              + "' due to I/O error: "
              + e.getMessage(),
          e,
          this,
          false);
    }
  }

  private void updateInputMtimeIfNeeded(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException {
    if (targetType != TargetType.FILESET) {
      return;
    }

    try {
      // Update the mtime of the target of the symlink to force downstream re-execution of actions.
      // This is needed because dependency checking of Fileset output trees is unsound (it's a
      // directory).
      // Note that utime() on a symlink actually changes the mtime of its target.
      Path linkPath = getOutputPath(actionExecutionContext);
      if (linkPath.exists()) {
        // -1L means "use the current time".
        linkPath.setLastModifiedTime(-1L);
      } else {
        // Should only happen if the Fileset included no links.
        actionExecutionContext.getExecRoot().getRelative(getInputPath()).createDirectory();
      }
    } catch (IOException e) {
      throw new ActionExecutionException("failed to touch symbolic link '"
          + Iterables.getOnlyElement(getOutputs()).prettyPrint()
          + "' to the '" + Iterables.getOnlyElement(getInputs()).prettyPrint()
          + "' due to I/O error: " + e.getMessage(), e, this, false);
    }
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
    return targetType == TargetType.EXECUTABLE ? "ExecutableSymlink" : "Symlink";
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

  @Override
  public boolean mayInsensitivelyPropagateInputs() {
    return true;
  }
}

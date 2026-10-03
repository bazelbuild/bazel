// Copyright 2026 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Action that <em>declares</em> its output to be a copy of its input: the same content,
 * re-addressed at the output's exec path, sharing the input's digest. Both file-to-file and
 * tree-to-tree (directory) copies are supported; the input and output must be the same kind.
 *
 * <p>When a lazy-staging layer is present (remote execution or an output service, signalled by a
 * non-null {@linkplain ActionExecutionContext#getActionFileSystem action filesystem}), the action
 * performs no spawn and writes nothing: it injects the output's metadata as a {@linkplain
 * FileArtifactValue#createForContentCopy content-copy} value (for a tree, a {@link
 * TreeArtifactValue} flagged {@linkplain TreeArtifactValue#isContentCopy content-copy}), pointing
 * at the input's terminal content and requesting content materialization. Deciding <em>how</em> to
 * place that content (hard link, copy, reflink, remote digest reuse) is the execution strategy's
 * job when it stages the artifact -- this action never makes that decision, exactly as path mapping
 * keeps laydown out of the action, and Bazel itself moves no bytes.
 *
 * <p>On a plain local build there is no such layer: the action then materializes real content at
 * the output's own path (copy-on-write/reflink where the filesystem supports it). This is the one
 * case where a copy may cost bytes; it is what gives a stable {@code realpath} within the consuming
 * tree (which tools such as Node.js require).
 *
 * <p>The metadata records the input's <em>terminal</em> resolved path, not the immediate input, so
 * a copy-of-a-copy always resolves to real bytes rather than a byteless intermediate.
 *
 * <p>Contrast {@link SymlinkAction}, which is realized as a followable symlink.
 */
public final class CopyAction extends AbstractAction {
  private static final String GUID = "6b4f2c11-0e7a-4d3f-9a2c-8b1d5e6f7a90";

  @Nullable private final String progressMessage;

  public static CopyAction create(
      ActionOwner owner, Artifact input, Artifact output, String progressMessage) {
    return new CopyAction(owner, input, output, progressMessage);
  }

  private CopyAction(
      ActionOwner owner, Artifact primaryInput, Artifact primaryOutput, String progressMessage) {
    super(
        owner,
        NestedSetBuilder.create(Order.STABLE_ORDER, checkNotNull(primaryInput)),
        ImmutableSet.of(primaryOutput));
    this.progressMessage = progressMessage;
  }

  @Override
  public ActionResult execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException {
    Artifact input = getPrimaryInput();
    Artifact output = getPrimaryOutput();
    try {
      if (output.isTreeArtifact()) {
        executeTree(actionExecutionContext, (SpecialArtifact) input, (SpecialArtifact) output);
      } else {
        executeFile(actionExecutionContext, input, output);
      }
    } catch (IOException e) {
      String message =
          String.format(
              "failed to copy '%s' to '%s': %s",
              input.getExecPathString(), output.getExecPathString(), e.getMessage());
      throw new ActionExecutionException(message, e, this, false, createDetailedExitCode(message));
    }
    return ActionResult.EMPTY;
  }

  private void executeFile(ActionExecutionContext ctx, Artifact input, Artifact output)
      throws IOException {
    FileArtifactValue inputMetadata =
        checkNotNull(
            ctx.getInputMetadataProvider().getInputMetadata(input),
            "missing metadata for %s",
            input);
    // Resolve to the terminal source: a copy-of-a-copy must point at real bytes, never at a
    // byteless intermediate.
    PathFragment terminal =
        inputMetadata.getResolvedPath() != null
            ? inputMetadata.getResolvedPath()
            : input.getPath().asFragment();

    if (ctx.getActionFileSystem() == null) {
      // No lazy-staging layer: materialize real content at the output's own path, using
      // copy-on-write/reflink where the filesystem supports it (Files.copy). The metadata store
      // reads the resulting file back.
      Path from = ctx.getExecRoot().getFileSystem().getPath(terminal);
      Path to = output.getPath();
      to.getParentDirectory().createDirectoryAndParents();
      FileSystemUtils.copyFile(from, to);
      return;
    }

    // A lazy-staging layer is present (remote execution / output service). Declaration only: the
    // executor materializes the content when it stages the output, so Bazel moves no bytes.
    ctx.getOutputMetadataStore()
        .injectFile(output, FileArtifactValue.createForContentCopy(inputMetadata, terminal));
  }

  private void executeTree(
      ActionExecutionContext ctx, SpecialArtifact input, SpecialArtifact output)
      throws IOException {
    TreeArtifactValue inputTree =
        checkNotNull(
            ctx.getInputMetadataProvider().getTreeMetadata(input),
            "missing tree metadata for %s",
            input);
    PathFragment terminal = inputTree.getResolvedPath().orElse(input.getPath().asFragment());

    if (ctx.getActionFileSystem() == null) {
      // No lazy-staging layer: recreate the directory tree with real content at the output's path.
      Path from = ctx.getExecRoot().getFileSystem().getPath(terminal);
      Path to = output.getPath();
      to.createDirectoryAndParents();
      FileSystemUtils.copyTreesBelow(from, to);
      return;
    }

    // Rebuild the tree value against the output tree artifact: the same children (re-parented) and
    // digests, flagged as a content copy resolving to the source tree root. The executor stages
    // each child by digest at the output's own path.
    TreeArtifactValue.Builder builder = TreeArtifactValue.newBuilder(output);
    for (Map.Entry<TreeFileArtifact, FileArtifactValue> child :
        inputTree.getChildValues().entrySet()) {
      builder.putChild(
          TreeFileArtifact.createTreeOutput(output, child.getKey().getParentRelativePath()),
          child.getValue());
    }
    builder.setResolvedPath(terminal);
    builder.setContentCopy(true);
    ctx.getOutputMetadataStore().injectTree(output, builder.build());
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable InputMetadataProvider inputMetadataProvider,
      Fingerprint fp) {
    fp.addString(GUID);
  }

  @Override
  protected String getRawProgressMessage() {
    return progressMessage;
  }

  @Override
  public String getMnemonic() {
    return "Copy";
  }

  @Override
  public boolean mayInsensitivelyPropagateInputs() {
    return true;
  }

  @Override
  public PlatformInfo getExecutionPlatform() {
    return PlatformInfo.EMPTY_PLATFORM_INFO;
  }

  @Override
  public ImmutableMap<String, String> getExecProperties() {
    return ImmutableMap.of();
  }

  private static DetailedExitCode createDetailedExitCode(String message) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setExecution(Execution.newBuilder().setCode(Execution.Code.SOURCE_INPUT_IO_EXCEPTION))
            .build());
  }
}

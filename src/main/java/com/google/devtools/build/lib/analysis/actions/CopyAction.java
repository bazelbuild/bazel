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

import com.google.common.annotations.VisibleForTesting;
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
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Action that makes the content of one artifact available at another path, without spawning.
 *
 * <p>Files copy to files and trees (directories) to trees. The output has the same digest as the
 * input but its own, independent metadata.
 *
 * <ul>
 *   <li>If the input is a generated file whose bytes are remote-only, no bytes move: the output
 *       gets a fresh remote metadata value with the input's digest and size. It is downloaded to
 *       the output's own path if a local consumer needs it. The value is never shared with the
 *       input (remote values are mutable) and carries no resolved path, so no symlink to the input
 *       is ever planted.
 *   <li>Otherwise (local bytes, source files, external repository files) the file is copied to the
 *       output's own path, using copy-on-write where the file system supports it. The output
 *       metadata is then read from the copy by the output metadata store, which also normalizes
 *       permissions. The source's digest is deliberately not reused: output permissions are not
 *       available to actions, and injected metadata skips the store's chmod.
 * </ul>
 *
 * <p>A tree copy is driven by the input tree's recorded children, so links or empty directories
 * that happen to be on disk below the input are never reproduced. Each child is decided on its own,
 * so a tree may mix remote and local files.
 *
 * <p>Like {@link SymlinkAction}, this action may pass its inputs through to its outputs, so that
 * rewinding of a lost input also rebuilds the input's producer.
 */
public final class CopyAction extends AbstractAction {
  private static final String GUID = "6b4f2c11-0e7a-4d3f-9a2c-8b1d5e6f7a90";

  @Nullable private final String progressMessage;

  public static CopyAction create(
      ActionOwner owner, Artifact input, Artifact output, @Nullable String progressMessage) {
    return new CopyAction(owner, input, output, progressMessage);
  }

  private CopyAction(
      ActionOwner owner,
      Artifact primaryInput,
      Artifact primaryOutput,
      @Nullable String progressMessage) {
    super(
        owner,
        NestedSetBuilder.create(Order.STABLE_ORDER, checkNotNull(primaryInput)),
        ImmutableSet.of(primaryOutput));
    this.progressMessage = progressMessage;
  }

  @Override
  public ActionResult execute(ActionExecutionContext ctx) throws ActionExecutionException {
    Artifact input = getPrimaryInput();
    Artifact output = getPrimaryOutput();
    try {
      if (output.isTreeArtifact()) {
        copyTree(ctx, (SpecialArtifact) input, (SpecialArtifact) output);
      } else {
        copyFile(ctx, input, output);
      }
    } catch (IOException e) {
      String message =
          String.format(
              "failed to copy '%s' to '%s': %s",
              input.getExecPathString(), output.getExecPathString(), e.getMessage());
      throw new ActionExecutionException(message, e, this, false, createDetailedExitCode(message));
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      String message = "interrupted while copying " + input.getExecPathString();
      throw new ActionExecutionException(message, e, this, false, createDetailedExitCode(message));
    }
    return ActionResult.EMPTY;
  }

  private void copyFile(ActionExecutionContext ctx, Artifact input, Artifact output)
      throws IOException {
    FileArtifactValue metadata =
        checkNotNull(
            ctx.getInputMetadataProvider().getInputMetadata(input),
            "missing metadata for %s",
            input);
    if (isRemoteOnly(input, metadata)) {
      ctx.getOutputMetadataStore().injectFile(output, freshRemoteCopy(metadata));
      return;
    }
    // The output metadata store reads (and chmods) the copy when the metadata is requested.
    copyLocalFile(ctx.getInputPath(input), output.getPath());
  }

  private void copyTree(ActionExecutionContext ctx, SpecialArtifact input, SpecialArtifact output)
      throws IOException, InterruptedException {
    TreeArtifactValue inputTree =
        checkNotNull(
            ctx.getInputMetadataProvider().getTreeMetadata(input),
            "missing tree metadata for %s",
            input);
    boolean anyRemote = false;
    boolean anyLocal = false;
    for (Map.Entry<TreeFileArtifact, FileArtifactValue> child :
        inputTree.getChildValues().entrySet()) {
      if (isRemoteOnly(input, child.getValue())) {
        anyRemote = true;
        continue;
      }
      anyLocal = true;
      TreeFileArtifact outChild =
          TreeFileArtifact.createTreeOutput(output, child.getKey().getParentRelativePath());
      copyLocalFile(ctx.getInputPath(child.getKey()), outChild.getPath());
    }
    if (!anyRemote) {
      // Nothing to inject: the output metadata store walks what was copied (and normalizes its
      // permissions). An empty input tree needs its (empty) output directory created here.
      if (!anyLocal) {
        output.getPath().forHostFileSystem().createDirectoryAndParents();
      }
      return;
    }

    // Mixed or fully remote tree. Have the store read and chmod the local copies first (a no-op if
    // there are none), then assemble the final value.
    TreeArtifactValue localTree =
        anyLocal ? ctx.getOutputMetadataStore().getTreeArtifactValue(output) : null;
    TreeArtifactValue.Builder builder = TreeArtifactValue.newBuilder(output);
    for (Map.Entry<TreeFileArtifact, FileArtifactValue> child :
        inputTree.getChildValues().entrySet()) {
      TreeFileArtifact outChild =
          TreeFileArtifact.createTreeOutput(output, child.getKey().getParentRelativePath());
      if (isRemoteOnly(input, child.getValue())) {
        builder.putChild(outChild, freshRemoteCopy(child.getValue()));
      } else {
        builder.putChild(
            outChild,
            FileArtifactValue.createForCopyOutput(
                checkNotNull(localTree).getChildValues().get(outChild)));
      }
    }
    ctx.getOutputMetadataStore().injectTree(output, builder.build());
  }

  private static boolean isRemoteOnly(Artifact input, FileArtifactValue metadata) {
    return !input.isSourceArtifact() && metadata.isRemote();
  }

  /**
   * Returns a new, independent remote value with the same digest and size. Remote values are
   * mutable (expiration, local materialization), so they must never be shared between artifacts.
   * No resolved path is set, as that would make the prefetcher plant a symlink to the input. The
   * value is marked as a copy output, so that no strategy stages it as a symlink.
   */
  private static FileArtifactValue freshRemoteCopy(FileArtifactValue source) {
    return FileArtifactValue.createForCopyOutput(
        FileArtifactValue.createForRemoteFileWithMaterializationData(
            source.getDigest(),
            source.getSize(),
            source.getLocationIndex(),
            source.getExpirationTime(),
            source.isInMemoryOutput()));
  }

  /**
   * Copies a local file. The source is read through the action's path resolver, so that files that
   * only exist in an overlay or action file system (e.g. repository contents served by a remote
   * repo contents cache, or in-memory inputs) can be read. When that is the native file system,
   * {@link FileSystemUtils#copyFile} clones. Otherwise it falls back to a stream copy. The
   * destination is always written on the host file system.
   */
  private static void copyLocalFile(Path from, Path to) throws IOException {
    copyFileWithFallback(from, to.forHostFileSystem(), FileSystemUtils::copyFile);
  }

  /** A copy primitive, for testing the fallback. */
  @VisibleForTesting
  interface Copier {
    void copy(Path from, Path to) throws IOException;
  }

  /**
   * Copies {@code from} to {@code to}, creating parent directories. If {@code primary} fails (e.g.
   * a file system that cannot clone and does not fall back), retries once with a stream copy. The
   * destination is never written in place, so a failed attempt leaves no partial output.
   */
  @VisibleForTesting
  static void copyFileWithFallback(Path from, Path to, Copier primary) throws IOException {
    to.getParentDirectory().createDirectoryAndParents();
    try {
      primary.copy(from, to);
      return;
    } catch (IOException e) {
      // Fall through to the plain copy below.
    }
    to.delete();
    try (InputStream in = from.getInputStream();
        OutputStream out = to.getOutputStream()) {
      in.transferTo(out);
    } catch (IOException e) {
      to.delete();
      throw e;
    }
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
  public boolean producesCopyOutputs() {
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
            .setExecution(Execution.newBuilder().setCode(Execution.Code.FILE_WRITE_IO_EXCEPTION))
            .build());
  }
}

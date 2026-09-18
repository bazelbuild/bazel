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
package com.google.devtools.build.lib.remote;

import static com.google.devtools.build.lib.remote.util.Futures.getFromFuture;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.RunningActionEvent;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.analysis.actions.FileWriteActionContext;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.Blob;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.util.DeterministicWriter;
import java.io.IOException;
import java.io.InputStream;
import java.time.Duration;
import java.time.Instant;

/**
 * A {@link FileWriteActionContext} that stores the contents of the file in the remote cache and
 * records them as a remote output instead of writing them to disk.
 *
 * <p>The file is written to disk instead if the file write isn't remotable, if the action doesn't
 * run on a {@link RemoteActionFileSystem}, if uploads to the remote cache are disabled or if the
 * output has to be downloaded anyway according to {@code --remote_download_outputs}. If the remote
 * cache loses the contents later, action rewinding re-executes the action to store them again.
 */
public final class RemoteFileWriteStrategy implements FileWriteActionContext {
  private final FileWriteActionContext localStrategy;
  private final CombinedCache combinedCache;
  private final RemoteOutputChecker remoteOutputChecker;
  private final DigestUtil digestUtil;
  private final String buildRequestId;
  private final String commandId;
  private final Duration remoteCacheTtl;
  private final boolean uploadEnabled;
  private final boolean verboseFailures;

  public RemoteFileWriteStrategy(
      FileWriteActionContext localStrategy,
      CombinedCache combinedCache,
      RemoteOutputChecker remoteOutputChecker,
      DigestUtil digestUtil,
      String buildRequestId,
      String commandId,
      Duration remoteCacheTtl,
      boolean uploadEnabled,
      boolean verboseFailures) {
    this.localStrategy = localStrategy;
    this.combinedCache = combinedCache;
    this.remoteOutputChecker = remoteOutputChecker;
    this.digestUtil = digestUtil;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.remoteCacheTtl = remoteCacheTtl;
    this.uploadEnabled = uploadEnabled;
    this.verboseFailures = verboseFailures;
  }

  @Override
  public ImmutableList<SpawnResult> writeOutputToFile(
      AbstractAction action,
      ActionExecutionContext actionExecutionContext,
      DeterministicWriter deterministicWriter,
      boolean makeExecutable,
      boolean isRemotable,
      Artifact output)
      throws InterruptedException, ExecException {
    // Non-remotable file writes are consumed by Bazel itself and thus have to exist locally.
    // Outputs that are requested for download would be materialized right after the action anyway.
    if (!isRemotable
        || !uploadEnabled
        || !(actionExecutionContext.getActionFileSystem()
            instanceof RemoteActionFileSystem remoteActionFileSystem)
        || remoteOutputChecker.shouldDownloadOutput(
            output.getExecPath(), /* treeRootExecPath= */ null)) {
      return localStrategy.writeOutputToFile(
          action, actionExecutionContext, deterministicWriter, makeExecutable, isRemotable, output);
    }

    actionExecutionContext.getEventHandler().post(new RunningActionEvent(action, "remote"));
    try (var _ = Profiler.instance().profile("RemoteFileWriteStrategy.writeOutputToFile")) {
      Digest digest;
      try {
        digest = digestUtil.compute(deterministicWriter);
      } catch (IOException e) {
        throw new EnvironmentalExecException(e, Code.FILE_WRITE_IO_EXCEPTION);
      }

      RemoteActionExecutionContext context =
          RemoteActionExecutionContext.create(buildRequestMetadata(action, actionExecutionContext));
      try {
        ImmutableSet<Digest> missingDigests =
            getFromFuture(combinedCache.findMissingDigests(context, ImmutableList.of(digest)));
        if (!missingDigests.isEmpty()) {
          getFromFuture(
              combinedCache.uploadBlob(
                  context, digest, new DeterministicWriterBlob(deterministicWriter, output)));
        }
      } catch (IOException e) {
        // Remote cache failures shouldn't fail the build, so write the file to disk instead.
        actionExecutionContext
            .getEventHandler()
            .handle(
                Event.warn(
                    "Remote Cache: " + Utils.grpcAwareErrorMessage(e, verboseFailures)));
        return localStrategy.writeOutputToFile(
            action,
            actionExecutionContext,
            deterministicWriter,
            makeExecutable,
            isRemotable,
            output);
      }

      // TODO: Bazel currently marks all output files as executable after local execution and
      // stages all files as executable for remote execution, so the executable bit isn't tracked
      // in the metadata yet.
      try {
        remoteActionFileSystem.injectRemoteFile(
            actionExecutionContext.getInputPath(output).asFragment(),
            DigestUtil.toBinaryDigest(digest),
            digest.getSizeBytes(),
            Instant.now().plus(remoteCacheTtl),
            /* inMemoryOutput= */ false);
      } catch (IOException e) {
        throw new EnvironmentalExecException(e, Code.FILE_WRITE_IO_EXCEPTION);
      }
    }
    return ImmutableList.of();
  }

  private RequestMetadata buildRequestMetadata(
      AbstractAction action, ActionExecutionContext actionExecutionContext)
      throws InterruptedException {
    ActionOwner owner = action.getOwner();
    return TracingMetadataUtils.buildMetadata(
        buildRequestId,
        commandId,
        action.getKey(
            actionExecutionContext.getActionKeyContext(),
            actionExecutionContext.getInputMetadataProvider()),
        action.getMnemonic(),
        owner.getLabel() != null ? owner.getLabel().getCanonicalForm() : null,
        owner.getConfigurationChecksum());
  }

  /** Streams the contents of a {@link DeterministicWriter} through a bounded pipe. */
  private record DeterministicWriterBlob(DeterministicWriter writer, Artifact output)
      implements Blob {
    @Override
    public InputStream get() {
      return writer.getInputStream(Chunker.getDefaultChunkSize());
    }

    @Override
    public String description() {
      return output.getExecPathString();
    }
  }
}

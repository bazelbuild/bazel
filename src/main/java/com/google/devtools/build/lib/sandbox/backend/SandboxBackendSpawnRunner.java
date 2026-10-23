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
package com.google.devtools.build.lib.sandbox.backend;

import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.ExecuteRequest;
import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.Platform;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnMetrics;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.analysis.platform.PlatformUtils;
import com.google.devtools.build.lib.exec.SpawnExecutingEvent;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.SpawnSchedulingEvent;
import com.google.devtools.build.lib.remote.common.ActionKey;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import com.google.devtools.build.lib.remote.merkletree.MerkleTreeComputer;
import com.google.devtools.build.lib.remote.merkletree.MerkleTreeComputer.BlobPolicy;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.rpc.Code;
import com.google.rpc.PreconditionFailure;
import io.grpc.StatusRuntimeException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.TreeSet;
import java.util.function.BooleanSupplier;
import javax.annotation.Nullable;

/**
 * Runs spawns in a sandbox backend.
 *
 * <p>The backend always executes the action. Its inputs are made known to the backend by uploading
 * the Merkle tree of the input root, in which files are only referred to by file URI, see {@link
 * FileUriBlobs}, and its outputs are brought back by {@link SandboxBackendOutputs}.
 */
public final class SandboxBackendSpawnRunner implements SpawnRunner {
  // The number of times a spawn is run again after the backend reported that it doesn't hold all
  // inputs, which means that it has evicted something since Bazel last asked.
  private static final int MAX_MISSING_RETRIES = 3;

  /** Provides the session to run spawns in. */
  @FunctionalInterface
  interface SessionProvider {
    SandboxBackendSession get() throws IOException, InterruptedException;
  }

  private final String name;
  private final SessionProvider sessionProvider;
  private final BooleanSupplier available;
  private final Path execRoot;
  private final Path scratchRoot;
  private final String workspaceName;
  private final String buildRequestId;
  private final String commandId;

  /**
   * @param name the name of the backend, which is also the name of the strategy
   * @param available whether the backend can be started at all, which includes that the operating
   *     system is supported, so that spawns are declined and run by another strategy if it can't
   * @param scratchRoot a directory on the file system of the exec root in which to assemble outputs
   */
  SandboxBackendSpawnRunner(
      String name,
      SessionProvider sessionProvider,
      BooleanSupplier available,
      Path execRoot,
      Path scratchRoot,
      String workspaceName,
      String buildRequestId,
      String commandId) {
    this.name = name;
    this.sessionProvider = sessionProvider;
    this.available = available;
    this.execRoot = execRoot;
    this.scratchRoot = scratchRoot;
    this.workspaceName = workspaceName;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
  }

  @Override
  public String getName() {
    return name;
  }

  @Override
  public boolean canExec(Spawn spawn) {
    return available.getAsBoolean() && Spawns.mayBeSandboxed(spawn);
  }

  @Override
  public boolean handlesCaching() {
    return false;
  }

  @Override
  public SpawnResult exec(Spawn spawn, SpawnExecutionContext context)
      throws ExecException, InterruptedException, IOException {
    Stopwatch total = Stopwatch.createStarted();
    // The backend binds inputs to files on this machine.
    context.prefetchInputsAndWait();

    SandboxBackendSession session;
    try {
      session = sessionProvider.get();
    } catch (IOException e) {
      return failure(context, e.getMessage(), total);
    }
    DigestUtil digestUtil = session.digestUtil();
    MerkleTreeComputer merkleTreeComputer =
        session.merkleTreeComputer(buildRequestId, commandId, workspaceName);
    RemotePathResolver resolver =
        RemotePathResolver.createMapped(
            RemotePathResolver.createDefault(execRoot), execRoot, spawn.getPathMapper());

    Platform platform = PlatformUtils.getPlatformProto(spawn, /* remoteOptions= */ null);
    Command command = buildCommand(spawn, resolver, platform);
    Digest commandDigest = digestUtil.compute(command);

    SpawnMetrics.Builder metrics = SpawnMetrics.Builder.forLocalExec();
    for (int attempt = 0; ; attempt++) {
      boolean force = attempt > 0;
      MerkleTree.Uploadable tree =
          (MerkleTree.Uploadable)
              merkleTreeComputer.buildForSpawn(
                  spawn,
                  /* toolInputs= */ ImmutableSet.of(),
                  /* scrubber= */ null,
                  context,
                  force ? BlobPolicy.KEEP_AND_REUPLOAD : BlobPolicy.KEEP);
      Action action =
          Utils.buildAction(
              commandDigest,
              tree.digest(),
              platform,
              context.getTimeout(),
              /* cacheable= */ false,
              /* salt= */ null);
      ActionKey actionKey = digestUtil.computeActionKey(action);
      context.setDigest(digestUtil.asSpawnLogProto(actionKey));
      RemoteActionExecutionContext remoteContext =
          RemoteActionExecutionContext.create(
              spawn, context, buildMetadata(spawn, actionKey.digest()));

      context.report(SpawnSchedulingEvent.create(name));
      Stopwatch uploadTime = Stopwatch.createStarted();
      try {
        session
            .cache()
            .ensureInputsPresent(
                remoteContext,
                tree,
                Map.of(actionKey.digest(), action, commandDigest, command),
                force);
      } catch (IOException e) {
        return failure(context, messageFor(session, e), total);
      }
      metrics.setUploadTime(uploadTime.elapsed());

      context.report(SpawnExecutingEvent.create(name));
      ExecuteResponse response;
      try {
        response =
            session.execute(
                remoteContext.getRequestMetadata(),
                ExecuteRequest.newBuilder()
                    .setSkipCacheLookup(true)
                    .setActionDigest(actionKey.digest())
                    .setDigestFunction(digestUtil.getDigestFunction())
                    .build());
      } catch (StatusRuntimeException e) {
        response = ExecuteResponse.newBuilder().setStatus(SandboxBackendSession.statusOf(e)).build();
      } catch (IOException e) {
        return failure(context, messageFor(session, e), total);
      }

      com.google.rpc.Status status = response.getStatus();
      if (status.getCode() == Code.OK_VALUE) {
        return finish(spawn, context, session, remoteContext, resolver, command, response, metrics, total);
      }
      if (isMissing(status) && attempt < MAX_MISSING_RETRIES) {
        continue;
      }
      if (status.getCode() == Code.DEADLINE_EXCEEDED_VALUE) {
        return resultBuilder(context, metrics, total)
            .setStatus(Status.TIMEOUT)
            .setExitCode(SpawnResult.POSIX_TIMEOUT_EXIT_CODE)
            .setFailureDetail(failureDetail("timed out", FailureDetails.Spawn.Code.TIMEOUT))
            .build();
      }
      return failure(
          context,
          messageFor(
              session,
              new IOException(
                  String.format("sandbox backend failed the action: %s", describe(status)))),
          total);
    }
  }

  private SpawnResult finish(
      Spawn spawn,
      SpawnExecutionContext context,
      SandboxBackendSession session,
      RemoteActionExecutionContext remoteContext,
      RemotePathResolver resolver,
      Command command,
      ExecuteResponse response,
      SpawnMetrics.Builder metrics,
      Stopwatch total)
      throws ExecException, InterruptedException, IOException {
    ActionResult result = response.getResult();
    Path scratch = scratchRoot.getRelative(Integer.toString(context.getId()));
    scratch.deleteTree();
    scratch.createDirectoryAndParents();
    try {
      Stopwatch fetchTime = Stopwatch.createStarted();
      SandboxBackendOutputs outputs = new SandboxBackendOutputs(session, scratch);
      Artifact stdoutOutput = spawn.getStdout();
      FileOutErr outErr = context.getFileOutErr();
      FileOutErr tmpOutErr = outErr.childOutErr();
      try {
        outputs.download(
            remoteContext,
            result,
            resolver,
            command.getOutputPathsList(),
            stdoutOutput != null ? stdoutOutput.getPath() : null);
        outputs.downloadOutErr(
            remoteContext,
            result,
            stdoutOutput == null ? tmpOutErr.getOutputStream() : null,
            tmpOutErr.getErrorStream());
        outputs.await();
      } catch (IOException e) {
        tmpOutErr.clearOut();
        tmpOutErr.clearErr();
        return failure(context, messageFor(session, e), total);
      }
      FileOutErr.dump(tmpOutErr, outErr);
      // Only one of the branches of a dynamically executed spawn gets to write outputs.
      context.lockOutputFiles(result.getExitCode(), response.getMessage(), tmpOutErr);
      tmpOutErr.clearOut();
      tmpOutErr.clearErr();
      outputs.commit();
      metrics.setFetchTime(fetchTime.elapsed());
    } finally {
      scratch.deleteTree();
    }
    SpawnResult.Builder builder =
        resultBuilder(context, metrics, total)
            .setStatus(result.getExitCode() == 0 ? Status.SUCCESS : Status.NON_ZERO_EXIT)
            .setExitCode(result.getExitCode());
    if (result.getExitCode() != 0) {
      builder.setFailureDetail(
          failureDetail("exited with code " + result.getExitCode(), FailureDetails.Spawn.Code.NON_ZERO_EXIT));
    }
    return builder.build();
  }

  private SpawnResult.Builder resultBuilder(
      SpawnExecutionContext context, SpawnMetrics.Builder metrics, Stopwatch total) {
    return getSpawnResultBuilder(context)
        .setSpawnMetrics(metrics.setTotalTime(total.elapsed()).build());
  }

  private static FailureDetail failureDetail(String message, FailureDetails.Spawn.Code code) {
    return FailureDetail.newBuilder()
        .setMessage(message)
        .setSpawn(FailureDetails.Spawn.newBuilder().setCode(code))
        .build();
  }

  private static String messageFor(SandboxBackendSession session, IOException cause) {
    return !session.isAlive() && session.deathMessage() != null
        ? session.deathMessage()
        : cause.getMessage();
  }

  private SpawnResult failure(SpawnExecutionContext context, String message, Stopwatch total) {
    return getSpawnResultBuilder(context)
        .setStatus(Status.EXECUTION_FAILED)
        .setExitCode(1)
        .setFailureMessage(message)
        .setFailureDetail(failureDetail(message, FailureDetails.Spawn.Code.EXEC_IO_EXCEPTION))
        .setSpawnMetrics(SpawnMetrics.Builder.forLocalExec().setTotalTime(total.elapsed()).build())
        .build();
  }

  private Command buildCommand(Spawn spawn, RemotePathResolver resolver, @Nullable Platform platform) {
    Command.Builder command = Command.newBuilder();
    Artifact stdout = spawn.getStdout();
    ArrayList<String> outputPaths = new ArrayList<>();
    for (ActionInput output : spawn.getOutputFiles()) {
      if (!output.equals(stdout)) {
        outputPaths.add(internalToUnicode(resolver.localPathToOutputPath(output)));
      }
    }
    command.addAllOutputPaths(Ordering.natural().sortedCopy(outputPaths));
    if (platform != null) {
      command.setPlatform(platform);
    }
    for (String argument : spawn.getArguments()) {
      command.addArguments(internalToUnicode(argument));
    }
    Map<String, String> environment = spawn.getEnvironment();
    for (String variable : new TreeSet<>(environment.keySet())) {
      command
          .addEnvironmentVariablesBuilder()
          .setName(internalToUnicode(variable))
          .setValue(internalToUnicode(environment.get(variable)));
    }
    return command.build();
  }

  private RequestMetadata buildMetadata(Spawn spawn, Digest actionDigest) {
    ActionExecutionMetadata owner = spawn.getResourceOwner();
    return TracingMetadataUtils.buildMetadata(
        buildRequestId,
        commandId,
        actionDigest.getHash(),
        owner != null ? owner.getMnemonic() : null,
        owner != null && owner.getOwner().getLabel() != null
            ? owner.getOwner().getLabel().getCanonicalForm()
            : null,
        owner != null ? owner.getOwner().getConfigurationChecksum() : null);
  }

  /** Whether the status reports that the backend doesn't hold some blobs. */
  private static boolean isMissing(com.google.rpc.Status status) {
    if (status.getCode() != Code.FAILED_PRECONDITION_VALUE) {
      return false;
    }
    for (Any detail : status.getDetailsList()) {
      if (!detail.is(PreconditionFailure.class)) {
        continue;
      }
      try {
        for (PreconditionFailure.Violation violation :
            detail.unpack(PreconditionFailure.class).getViolationsList()) {
          if (violation.getType().equals("MISSING")) {
            return true;
          }
        }
      } catch (InvalidProtocolBufferException e) {
        return false;
      }
    }
    return false;
  }

  private static String describe(com.google.rpc.Status status) {
    return Code.forNumber(status.getCode()) + ": " + status.getMessage();
  }
}

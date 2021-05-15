// Copyright 2021 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;
import static com.google.devtools.build.lib.remote.util.Utils.getInMemoryOutputPath;
import static com.google.devtools.build.lib.remote.util.Utils.hasFilesToDownload;
import static com.google.devtools.build.lib.remote.util.Utils.shouldDownloadAllSpawnOutputs;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.ExecuteRequest;
import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.ExecutedActionMetadata;
import build.bazel.remote.execution.v2.LogFile;
import build.bazel.remote.execution.v2.Platform;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.platform.PlatformUtils;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.remote.common.NetworkTime;
import com.google.devtools.build.lib.remote.common.OperationObserver;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.common.RemoteExecutionClient;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.remote.util.Utils.InMemoryOutput;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.Message;
import io.grpc.Status.Code;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeSet;
import javax.annotation.Nullable;

/**
 * A layer between spawn execution and remote execution exposing primitive operations for remote
 * cache and execution with spawn specific types.
 */
public class RemoteExecutionService {
  private final Path execRoot;
  private final RemotePathResolver remotePathResolver;
  private final String buildRequestId;
  private final String commandId;
  private final DigestUtil digestUtil;
  private final RemoteOptions remoteOptions;
  private final RemoteCache remoteCache;
  @Nullable private final RemoteExecutionClient remoteExecutor;
  private final ImmutableSet<ActionInput> filesToDownload;

  public RemoteExecutionService(
      Path execRoot,
      RemotePathResolver remotePathResolver,
      String buildRequestId,
      String commandId,
      DigestUtil digestUtil,
      RemoteOptions remoteOptions,
      RemoteCache remoteCache,
      @Nullable RemoteExecutionClient remoteExecutor,
      ImmutableSet<ActionInput> filesToDownload) {
    this.execRoot = execRoot;
    this.remotePathResolver = remotePathResolver;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.digestUtil = digestUtil;
    this.remoteOptions = remoteOptions;
    this.remoteCache = remoteCache;
    this.remoteExecutor = remoteExecutor;
    this.filesToDownload = filesToDownload;
  }

  static Command buildCommand(
      Collection<? extends ActionInput> outputs,
      List<String> arguments,
      ImmutableMap<String, String> env,
      @Nullable Platform platform,
      RemotePathResolver remotePathResolver) {
    Command.Builder command = Command.newBuilder();
    ArrayList<String> outputFiles = new ArrayList<>();
    ArrayList<String> outputDirectories = new ArrayList<>();
    for (ActionInput output : outputs) {
      String pathString = remotePathResolver.localPathToOutputPath(output);
      if (output instanceof Artifact && ((Artifact) output).isTreeArtifact()) {
        outputDirectories.add(pathString);
      } else {
        outputFiles.add(pathString);
      }
    }
    Collections.sort(outputFiles);
    Collections.sort(outputDirectories);
    command.addAllOutputFiles(outputFiles);
    command.addAllOutputDirectories(outputDirectories);

    if (platform != null) {
      command.setPlatform(platform);
    }
    command.addAllArguments(arguments);
    // Sorting the environment pairs by variable name.
    TreeSet<String> variables = new TreeSet<>(env.keySet());
    for (String var : variables) {
      command.addEnvironmentVariablesBuilder().setName(var).setValue(env.get(var));
    }

    String workingDirectory = remotePathResolver.getWorkingDirectory();
    if (!Strings.isNullOrEmpty(workingDirectory)) {
      command.setWorkingDirectory(workingDirectory);
    }
    return command.build();
  }

  /** A value class representing an action which can be executed remotely. */
  public static class RemoteAction {
    private final Spawn spawn;
    private final SpawnExecutionContext spawnExecutionContext;
    private final RemoteActionExecutionContext remoteActionExecutionContext;
    private final SortedMap<PathFragment, ActionInput> inputMap;
    private final MerkleTree merkleTree;
    private final Digest commandHash;
    private final Command command;
    private final Action action;
    private final ActionKey actionKey;

    RemoteAction(
        Spawn spawn,
        SpawnExecutionContext spawnExecutionContext,
        RemoteActionExecutionContext remoteActionExecutionContext,
        SortedMap<PathFragment, ActionInput> inputMap,
        MerkleTree merkleTree,
        Digest commandHash,
        Command command,
        Action action,
        ActionKey actionKey) {
      this.spawn = spawn;
      this.spawnExecutionContext = spawnExecutionContext;
      this.remoteActionExecutionContext = remoteActionExecutionContext;
      this.inputMap = inputMap;
      this.merkleTree = merkleTree;
      this.commandHash = commandHash;
      this.command = command;
      this.action = action;
      this.actionKey = actionKey;
    }

    /**
     * Returns the sum of file sizes plus protobuf sizes used to represent the inputs of this
     * action.
     */
    public long getInputBytes() {
      return merkleTree.getInputBytes();
    }

    /** Returns the number of input files of this action. */
    public long getInputFiles() {
      return merkleTree.getInputFiles();
    }

    /** Returns the id this is action. */
    public String getActionId() {
      return actionKey.getDigest().getHash();
    }

    /**
     * Returns a {@link SortedMap} which maps from input paths for remote action to {@link
     * ActionInput}.
     */
    public SortedMap<PathFragment, ActionInput> getInputMap() {
      return inputMap;
    }

    /**
     * Returns the {@link NetworkTime} instance used to measure the network time during the action
     * execution.
     */
    public NetworkTime getNetworkTime() {
      return remoteActionExecutionContext.getNetworkTime();
    }
  }

  /** Creates a new {@link RemoteAction} instance from spawn. */
  public RemoteAction buildRemoteAction(Spawn spawn, SpawnExecutionContext context)
      throws IOException, UserExecException {
    SortedMap<PathFragment, ActionInput> inputMap = remotePathResolver.getInputMapping(context);
    final MerkleTree merkleTree =
        MerkleTree.build(inputMap, context.getMetadataProvider(), execRoot, digestUtil);

    // Get the remote platform properties.
    Platform platform = PlatformUtils.getPlatformProto(spawn, remoteOptions);

    Command command =
        buildCommand(
            spawn.getOutputFiles(),
            spawn.getArguments(),
            spawn.getEnvironment(),
            platform,
            remotePathResolver);
    Digest commandHash = digestUtil.compute(command);
    Action action =
        Utils.buildAction(
            commandHash,
            merkleTree.getRootDigest(),
            platform,
            context.getTimeout(),
            Spawns.mayBeCachedRemotely(spawn));

    ActionKey actionKey = digestUtil.computeActionKey(action);

    RequestMetadata metadata =
        TracingMetadataUtils.buildMetadata(
            buildRequestId, commandId, actionKey.getDigest().getHash(), spawn.getResourceOwner());
    RemoteActionExecutionContext remoteActionExecutionContext =
        RemoteActionExecutionContext.create(metadata);

    return new RemoteAction(
        spawn,
        context,
        remoteActionExecutionContext,
        inputMap,
        merkleTree,
        commandHash,
        command,
        action,
        actionKey);
  }

  /** A value class representing the result of remotely executed {@link RemoteAction}. */
  public static class RemoteActionResult {
    private final ActionResult actionResult;
    @Nullable private final ExecuteResponse executeResponse;

    /** Creates a new {@link RemoteActionResult} instance from a cached result. */
    public static RemoteActionResult createFromCache(ActionResult cachedActionResult) {
      checkArgument(cachedActionResult != null, "cachedActionResult is null");
      return new RemoteActionResult(cachedActionResult, null);
    }

    /** Creates a new {@link RemoteActionResult} instance from a execute response. */
    public static RemoteActionResult createFromResponse(ExecuteResponse response) {
      checkArgument(response.hasResult(), "response doesn't have result");
      return new RemoteActionResult(response.getResult(), response);
    }

    public RemoteActionResult(
        ActionResult actionResult, @Nullable ExecuteResponse executeResponse) {
      this.actionResult = actionResult;
      this.executeResponse = executeResponse;
    }

    /** Returns the exit code of remote executed action. */
    public int getExitCode() {
      return actionResult.getExitCode();
    }

    /**
     * Returns the freeform informational message with details on the execution of the action that
     * may be displayed to the user upon failure or when requested explicitly.
     */
    public String getMessage() {
      return executeResponse != null ? executeResponse.getMessage() : "";
    }

    /** Returns the details of the execution that originally produced this result. */
    public ExecutedActionMetadata getExecutionMetadata() {
      return actionResult.getExecutionMetadata();
    }

    /** Returns whether the action is executed successfully. */
    public boolean success() {
      if (executeResponse != null) {
        if (executeResponse.getStatus().getCode() != Code.OK.value()) {
          return false;
        }
      }

      return actionResult.getExitCode() == 0;
    }

    /** Returns {@code true} if this result is from a cache. */
    public boolean cacheHit() {
      if (executeResponse == null) {
        return true;
      }

      return executeResponse.getCachedResult();
    }

    /**
     * Returns the underlying {@link ExecuteResponse} or {@code null} if this result is from a
     * cache.
     */
    @Nullable
    public ExecuteResponse getResponse() {
      return executeResponse;
    }
  }

  /** Lookup the remote cache for the given {@link RemoteAction}. {@code null} if not found. */
  @Nullable
  public RemoteActionResult lookupCache(RemoteAction action)
      throws IOException, InterruptedException {
    ActionResult actionResult =
        remoteCache.downloadActionResult(
            action.remoteActionExecutionContext, action.actionKey, /* inlineOutErr= */ false);

    if (actionResult == null) {
      return null;
    }

    return RemoteActionResult.createFromCache(actionResult);
  }

  /** Downloads outputs of a remotely executed action from remote cache. */
  @Nullable
  public InMemoryOutput downloadOutputs(RemoteAction action, RemoteActionResult result)
      throws InterruptedException, IOException, ExecException {
    RemoteOutputsMode remoteOutputsMode = remoteOptions.remoteOutputsMode;
    boolean downloadOutputs =
        shouldDownloadAllSpawnOutputs(
            remoteOutputsMode,
            /* exitCode = */ result.actionResult.getExitCode(),
            hasFilesToDownload(action.spawn.getOutputFiles(), filesToDownload));
    InMemoryOutput inMemoryOutput = null;
    if (downloadOutputs) {
      remoteCache.download(
          action.remoteActionExecutionContext,
          remotePathResolver,
          result.actionResult,
          action.spawnExecutionContext.getFileOutErr(),
          action.spawnExecutionContext::lockOutputFiles);
    } else {
      PathFragment inMemoryOutputPath = getInMemoryOutputPath(action.spawn);
      inMemoryOutput =
          remoteCache.downloadMinimal(
              action.remoteActionExecutionContext,
              remotePathResolver,
              result.actionResult,
              action.spawn.getOutputFiles(),
              inMemoryOutputPath,
              action.spawnExecutionContext.getFileOutErr(),
              action.spawnExecutionContext.getMetadataInjector(),
              action.spawnExecutionContext::lockOutputFiles);
    }

    return inMemoryOutput;
  }

  /** Upload outputs of a remote action which was executed locally to remote cache. */
  public void uploadOutputs(RemoteAction action)
      throws InterruptedException, IOException, ExecException {
    Collection<Path> outputFiles =
        action.spawn.getOutputFiles().stream()
            .map((inp) -> execRoot.getRelative(inp.getExecPath()))
            .collect(ImmutableList.toImmutableList());
    remoteCache.upload(
        action.remoteActionExecutionContext,
        remotePathResolver,
        action.actionKey,
        action.action,
        action.command,
        outputFiles,
        action.spawnExecutionContext.getFileOutErr());
  }

  /**
   * Upload inputs of a remote action to remote cache if they are not presented already.
   *
   * <p>Must be called before calling {@link #execute}.
   */
  public void uploadInputsIfNotPresent(RemoteAction action)
      throws IOException, InterruptedException {
    Preconditions.checkState(remoteCache instanceof RemoteExecutionCache);
    RemoteExecutionCache remoteExecutionCache = (RemoteExecutionCache) remoteCache;
    // Upload the command and all the inputs into the remote cache.
    Map<Digest, Message> additionalInputs = Maps.newHashMapWithExpectedSize(2);
    additionalInputs.put(action.actionKey.getDigest(), action.action);
    additionalInputs.put(action.commandHash, action.command);
    remoteExecutionCache.ensureInputsPresent(
        action.remoteActionExecutionContext, action.merkleTree, additionalInputs);
  }

  /**
   * Executes the remote action remotely and returns the result.
   *
   * @param acceptCachedResult tells remote execution server whether it should used cached result.
   * @param observer receives status updates during the execution.
   */
  public RemoteActionResult execute(
      RemoteAction action, boolean acceptCachedResult, OperationObserver observer)
      throws IOException, InterruptedException {
    Preconditions.checkNotNull(remoteExecutor, "remoteExecutor");

    ExecuteRequest.Builder requestBuilder =
        ExecuteRequest.newBuilder()
            .setInstanceName(remoteOptions.remoteInstanceName)
            .setActionDigest(action.actionKey.getDigest())
            .setSkipCacheLookup(!acceptCachedResult);
    if (remoteOptions.remoteResultCachePriority != 0) {
      requestBuilder
          .getResultsCachePolicyBuilder()
          .setPriority(remoteOptions.remoteResultCachePriority);
    }
    if (remoteOptions.remoteExecutionPriority != 0) {
      requestBuilder.getExecutionPolicyBuilder().setPriority(remoteOptions.remoteExecutionPriority);
    }

    ExecuteRequest request = requestBuilder.build();

    ExecuteResponse reply =
        remoteExecutor.executeRemotely(action.remoteActionExecutionContext, request, observer);

    return RemoteActionResult.createFromResponse(reply);
  }

  /** A value classes representing downloaded server logs. */
  public static class ServerLogs {
    public int logCount;
    public Path directory;
    @Nullable public Path lastLogPath;
  }

  /** Downloads server logs from a remotely executed action if any. */
  public ServerLogs maybeDownloadServerLogs(RemoteAction action, ExecuteResponse resp, Path logDir)
      throws InterruptedException, IOException {
    ServerLogs serverLogs = new ServerLogs();
    serverLogs.directory = logDir.getRelative(action.getActionId());

    ActionResult actionResult = resp.getResult();
    if (resp.getServerLogsCount() > 0
        && (actionResult.getExitCode() != 0 || resp.getStatus().getCode() != Code.OK.value())) {
      for (Map.Entry<String, LogFile> e : resp.getServerLogsMap().entrySet()) {
        if (e.getValue().getHumanReadable()) {
          serverLogs.lastLogPath = serverLogs.directory.getRelative(e.getKey());
          serverLogs.logCount++;
          getFromFuture(
              remoteCache.downloadFile(
                  action.remoteActionExecutionContext,
                  serverLogs.lastLogPath,
                  e.getValue().getDigest()));
        }
      }
    }

    return serverLogs;
  }
}

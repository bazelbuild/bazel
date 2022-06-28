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
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.base.Strings.isNullOrEmpty;
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.Futures.transform;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.RemoteCache.createFailureDetail;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;
import static com.google.devtools.build.lib.remote.util.Utils.getInMemoryOutputPath;
import static com.google.devtools.build.lib.remote.util.Utils.grpcAwareErrorMessage;
import static com.google.devtools.build.lib.remote.util.Utils.hasFilesToDownload;
import static com.google.devtools.build.lib.remote.util.Utils.shouldAcceptCachedResultFromCombinedCache;
import static com.google.devtools.build.lib.remote.util.Utils.shouldAcceptCachedResultFromDiskCache;
import static com.google.devtools.build.lib.remote.util.Utils.shouldAcceptCachedResultFromRemoteCache;
import static com.google.devtools.build.lib.remote.util.Utils.shouldDownloadAllSpawnOutputs;
import static com.google.devtools.build.lib.remote.util.Utils.shouldUploadLocalResultsToCombinedDisk;
import static com.google.devtools.build.lib.remote.util.Utils.shouldUploadLocalResultsToDiskCache;
import static com.google.devtools.build.lib.remote.util.Utils.shouldUploadLocalResultsToRemoteCache;
import static com.google.devtools.build.lib.remote.util.Utils.waitForBulkTransfer;
import static com.google.devtools.build.lib.util.StringUtil.decodeBytestringUtf8;
import static com.google.devtools.build.lib.util.StringUtil.encodeBytestringUtf8;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.ExecuteRequest;
import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.ExecutedActionMetadata;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.LogFile;
import build.bazel.remote.execution.v2.OutputDirectory;
import build.bazel.remote.execution.v2.OutputFile;
import build.bazel.remote.execution.v2.OutputSymlink;
import build.bazel.remote.execution.v2.Platform;
import build.bazel.remote.execution.v2.RequestMetadata;
import build.bazel.remote.execution.v2.SymlinkNode;
import build.bazel.remote.execution.v2.Tree;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.eventbus.Subscribe;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.ForbiddenActionInputException;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.actions.cache.MetadataInjector;
import com.google.devtools.build.lib.analysis.platform.PlatformUtils;
import com.google.devtools.build.lib.buildtool.buildevent.BuildInterruptedEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.SpawnInputExpander.InputWalker;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.RemoteExecutionService.ActionResultMetadata.DirectoryMetadata;
import com.google.devtools.build.lib.remote.RemoteExecutionService.ActionResultMetadata.FileMetadata;
import com.google.devtools.build.lib.remote.RemoteExecutionService.ActionResultMetadata.SymlinkMetadata;
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import com.google.devtools.build.lib.remote.common.OperationObserver;
import com.google.devtools.build.lib.remote.common.OutputDigestMismatchException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext.Step;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.CachedActionResult;
import com.google.devtools.build.lib.remote.common.RemoteExecutionClient;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.remote.util.Utils.InMemoryOutput;
import com.google.devtools.build.lib.server.FailureDetails.RemoteExecution;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import com.google.protobuf.ExtensionRegistry;
import com.google.protobuf.Message;
import io.grpc.Status.Code;
import io.reactivex.rxjava3.annotations.NonNull;
import io.reactivex.rxjava3.core.Scheduler;
import io.reactivex.rxjava3.core.Single;
import io.reactivex.rxjava3.core.SingleObserver;
import io.reactivex.rxjava3.disposables.Disposable;
import io.reactivex.rxjava3.schedulers.Schedulers;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Executor;
import java.util.concurrent.Phaser;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/**
 * A layer between spawn execution and remote execution exposing primitive operations for remote
 * cache and execution with spawn specific types.
 */
public class RemoteExecutionService {
  private final Reporter reporter;
  private final boolean verboseFailures;
  private final Path execRoot;
  private final RemotePathResolver remotePathResolver;
  private final String buildRequestId;
  private final String commandId;
  private final DigestUtil digestUtil;
  private final RemoteOptions remoteOptions;
  @Nullable private final RemoteCache remoteCache;
  @Nullable private final RemoteExecutionClient remoteExecutor;
  private final ImmutableSet<PathFragment> filesToDownload;
  @Nullable private final Path captureCorruptedOutputsDir;
  private final Cache<Object, MerkleTree> merkleTreeCache;
  private final Set<String> reportedErrors = new HashSet<>();
  private final Phaser backgroundTaskPhaser = new Phaser(1);

  private final Scheduler scheduler;

  private final AtomicBoolean shutdown = new AtomicBoolean(false);
  private final AtomicBoolean buildInterrupted = new AtomicBoolean(false);

  public RemoteExecutionService(
      Executor executor,
      Reporter reporter,
      boolean verboseFailures,
      Path execRoot,
      RemotePathResolver remotePathResolver,
      String buildRequestId,
      String commandId,
      DigestUtil digestUtil,
      RemoteOptions remoteOptions,
      @Nullable RemoteCache remoteCache,
      @Nullable RemoteExecutionClient remoteExecutor,
      ImmutableSet<ActionInput> filesToDownload,
      @Nullable Path captureCorruptedOutputsDir) {
    this.reporter = reporter;
    this.verboseFailures = verboseFailures;
    this.execRoot = execRoot;
    this.remotePathResolver = remotePathResolver;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.digestUtil = digestUtil;
    this.remoteOptions = remoteOptions;
    this.remoteCache = remoteCache;
    this.remoteExecutor = remoteExecutor;

    Caffeine<Object, Object> merkleTreeCacheBuilder = Caffeine.newBuilder().softValues();
    // remoteMerkleTreesCacheSize = 0 means limitless.
    if (remoteOptions.remoteMerkleTreeCacheSize != 0) {
      merkleTreeCacheBuilder.maximumSize(remoteOptions.remoteMerkleTreeCacheSize);
    }
    this.merkleTreeCache = merkleTreeCacheBuilder.build();

    ImmutableSet.Builder<PathFragment> filesToDownloadBuilder = ImmutableSet.builder();
    for (ActionInput actionInput : filesToDownload) {
      filesToDownloadBuilder.add(actionInput.getExecPath());
    }
    this.filesToDownload = filesToDownloadBuilder.build();
    this.captureCorruptedOutputsDir = captureCorruptedOutputsDir;

    this.scheduler = Schedulers.from(executor, /*interruptibleWorker=*/ true);
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
      String pathString = decodeBytestringUtf8(remotePathResolver.localPathToOutputPath(output));
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
    for (String arg : arguments) {
      command.addArguments(decodeBytestringUtf8(arg));
    }
    // Sorting the environment pairs by variable name.
    TreeSet<String> variables = new TreeSet<>(env.keySet());
    for (String var : variables) {
      command
          .addEnvironmentVariablesBuilder()
          .setName(decodeBytestringUtf8(var))
          .setValue(decodeBytestringUtf8(env.get(var)));
    }

    String workingDirectory = remotePathResolver.getWorkingDirectory();
    if (!Strings.isNullOrEmpty(workingDirectory)) {
      command.setWorkingDirectory(decodeBytestringUtf8(workingDirectory));
    }
    return command.build();
  }

  private static boolean useRemoteCache(RemoteOptions options) {
    return !isNullOrEmpty(options.remoteCache) || !isNullOrEmpty(options.remoteExecutor);
  }

  private static boolean useDiskCache(RemoteOptions options) {
    return options.diskCache != null && !options.diskCache.isEmpty();
  }

  /** Returns {@code true} if the {@code spawn} should accept cached results from remote cache. */
  public boolean shouldAcceptCachedResult(Spawn spawn) {
    if (remoteCache == null) {
      return false;
    }

    if (useRemoteCache(remoteOptions)) {
      if (useDiskCache(remoteOptions)) {
        return shouldAcceptCachedResultFromCombinedCache(remoteOptions, spawn);
      } else {
        return shouldAcceptCachedResultFromRemoteCache(remoteOptions, spawn);
      }
    } else {
      return shouldAcceptCachedResultFromDiskCache(remoteOptions, spawn);
    }
  }

  /**
   * Returns {@code true} if the local results of the {@code spawn} should be uploaded to remote
   * cache.
   */
  public boolean shouldUploadLocalResults(Spawn spawn) {
    if (remoteCache == null) {
      return false;
    }

    if (useRemoteCache(remoteOptions)) {
      if (useDiskCache(remoteOptions)) {
        return shouldUploadLocalResultsToCombinedDisk(remoteOptions, spawn);
      } else {
        return shouldUploadLocalResultsToRemoteCache(remoteOptions, spawn);
      }
    } else {
      return shouldUploadLocalResultsToDiskCache(remoteOptions, spawn);
    }
  }

  /** Returns {@code true} if the spawn may be executed remotely. */
  public boolean mayBeExecutedRemotely(Spawn spawn) {
    return remoteCache instanceof RemoteExecutionCache
        && remoteExecutor != null
        && Spawns.mayBeExecutedRemotely(spawn);
  }

  private SortedMap<PathFragment, ActionInput> buildOutputDirMap(Spawn spawn) {
    TreeMap<PathFragment, ActionInput> outputDirMap = new TreeMap<>();
    for (ActionInput output : spawn.getOutputFiles()) {
      if (output instanceof Artifact && ((Artifact) output).isTreeArtifact()) {
        outputDirMap.put(
            PathFragment.create(remotePathResolver.getWorkingDirectory())
                .getRelative(remotePathResolver.localPathToOutputPath(output.getExecPath())),
            output);
      }
    }
    return outputDirMap;
  }

  private MerkleTree buildInputMerkleTree(Spawn spawn, SpawnExecutionContext context)
      throws IOException, ForbiddenActionInputException {
    // Add output directories to inputs so that they are created as empty directories by the
    // executor. The spec only requires the executor to create the parent directory of an output
    // directory, which differs from the behavior of both local and sandboxed execution.
    SortedMap<PathFragment, ActionInput> outputDirMap = buildOutputDirMap(spawn);
    if (remoteOptions.remoteMerkleTreeCache) {
      MetadataProvider metadataProvider = context.getMetadataProvider();
      ConcurrentLinkedQueue<MerkleTree> subMerkleTrees = new ConcurrentLinkedQueue<>();
      remotePathResolver.walkInputs(
          spawn,
          context,
          (Object nodeKey, InputWalker walker) -> {
            subMerkleTrees.add(buildMerkleTreeVisitor(nodeKey, walker, metadataProvider));
          });
      if (!outputDirMap.isEmpty()) {
        subMerkleTrees.add(MerkleTree.build(outputDirMap, metadataProvider, execRoot, digestUtil));
      }
      return MerkleTree.merge(subMerkleTrees, digestUtil);
    } else {
      SortedMap<PathFragment, ActionInput> inputMap = remotePathResolver.getInputMapping(context);
      if (!outputDirMap.isEmpty()) {
        // The map returned by getInputMapping is mutable, but must not be mutated here as it is
        // shared with all other strategies.
        SortedMap<PathFragment, ActionInput> newInputMap = new TreeMap<>();
        newInputMap.putAll(inputMap);
        newInputMap.putAll(outputDirMap);
        inputMap = newInputMap;
      }
      return MerkleTree.build(inputMap, context.getMetadataProvider(), execRoot, digestUtil);
    }
  }

  private MerkleTree buildMerkleTreeVisitor(
      Object nodeKey, InputWalker walker, MetadataProvider metadataProvider)
      throws IOException, ForbiddenActionInputException {
    MerkleTree result = merkleTreeCache.getIfPresent(nodeKey);
    if (result == null) {
      result = uncachedBuildMerkleTreeVisitor(walker, metadataProvider);
      merkleTreeCache.put(nodeKey, result);
    }
    return result;
  }

  @VisibleForTesting
  public MerkleTree uncachedBuildMerkleTreeVisitor(
      InputWalker walker, MetadataProvider metadataProvider)
      throws IOException, ForbiddenActionInputException {
    ConcurrentLinkedQueue<MerkleTree> subMerkleTrees = new ConcurrentLinkedQueue<>();
    subMerkleTrees.add(
        MerkleTree.build(walker.getLeavesInputMapping(), metadataProvider, execRoot, digestUtil));
    walker.visitNonLeaves(
        (Object subNodeKey, InputWalker subWalker) -> {
          subMerkleTrees.add(buildMerkleTreeVisitor(subNodeKey, subWalker, metadataProvider));
        });
    return MerkleTree.merge(subMerkleTrees, digestUtil);
  }

  @Nullable
  private static ByteString buildSalt(Spawn spawn) {
    String workspace =
        spawn.getExecutionInfo().get(ExecutionRequirements.DIFFERENTIATE_WORKSPACE_CACHE);
    if (workspace != null) {
      Platform platform =
          Platform.newBuilder()
              .addProperties(
                  Platform.Property.newBuilder().setName("workspace").setValue(workspace).build())
              .build();
      return platform.toByteString();
    }

    return null;
  }

  /** Creates a new {@link RemoteAction} instance from spawn. */
  public RemoteAction buildRemoteAction(Spawn spawn, SpawnExecutionContext context)
      throws IOException, UserExecException, ForbiddenActionInputException {
    final MerkleTree merkleTree = buildInputMerkleTree(spawn, context);

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
            Spawns.mayBeCachedRemotely(spawn),
            buildSalt(spawn));

    ActionKey actionKey = digestUtil.computeActionKey(action);

    RequestMetadata metadata =
        TracingMetadataUtils.buildMetadata(
            buildRequestId, commandId, actionKey.getDigest().getHash(), spawn.getResourceOwner());
    RemoteActionExecutionContext remoteActionExecutionContext =
        RemoteActionExecutionContext.create(spawn, metadata);

    return new RemoteAction(
        spawn,
        context,
        remoteActionExecutionContext,
        remotePathResolver,
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
    @Nullable private final String cacheName;

    /** Creates a new {@link RemoteActionResult} instance from a cached result. */
    public static RemoteActionResult createFromCache(CachedActionResult cachedActionResult) {
      checkArgument(cachedActionResult != null, "cachedActionResult is null");
      return new RemoteActionResult(
          cachedActionResult.actionResult(), null, cachedActionResult.cacheName());
    }

    /** Creates a new {@link RemoteActionResult} instance from a execute response. */
    public static RemoteActionResult createFromResponse(ExecuteResponse response) {
      checkArgument(response.hasResult(), "response doesn't have result");
      return new RemoteActionResult(response.getResult(), response, /* cacheName */ null);
    }

    public RemoteActionResult(
        ActionResult actionResult,
        @Nullable ExecuteResponse executeResponse,
        @Nullable String cacheName) {
      this.actionResult = actionResult;
      this.executeResponse = executeResponse;
      this.cacheName = cacheName;
    }

    /** Returns the exit code of remote executed action. */
    public int getExitCode() {
      return actionResult.getExitCode();
    }

    public List<OutputFile> getOutputFiles() {
      return actionResult.getOutputFilesList();
    }

    public List<OutputSymlink> getOutputFileSymlinks() {
      return actionResult.getOutputFileSymlinksList();
    }

    public List<OutputDirectory> getOutputDirectories() {
      return actionResult.getOutputDirectoriesList();
    }

    public int getOutputDirectoriesCount() {
      return actionResult.getOutputDirectoriesCount();
    }

    public List<OutputSymlink> getOutputDirectorySymlinks() {
      return actionResult.getOutputDirectorySymlinksList();
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

    /** Returns cache name (disk/remote) when {@code cacheHit()} or {@code null} when not */
    @Nullable
    public String cacheName() {
      return cacheName;
    }

    /**
     * Returns the underlying {@link ExecuteResponse} or {@code null} if this result is from a
     * cache.
     */
    @Nullable
    public ExecuteResponse getResponse() {
      return executeResponse;
    }

    @Override
    public boolean equals(Object object) {
      if (!(object instanceof RemoteActionResult)) {
        return false;
      }

      RemoteActionResult that = (RemoteActionResult) object;
      return Objects.equals(actionResult, that.actionResult)
          && Objects.equals(executeResponse, that.executeResponse);
    }

    @Override
    public int hashCode() {
      return Objects.hash(actionResult, executeResponse);
    }
  }

  /** Lookup the remote cache for the given {@link RemoteAction}. {@code null} if not found. */
  @Nullable
  public RemoteActionResult lookupCache(RemoteAction action)
      throws IOException, InterruptedException {
    checkState(shouldAcceptCachedResult(action.getSpawn()), "spawn doesn't accept cached result");

    action.getRemoteActionExecutionContext().setStep(Step.CHECK_ACTION_CACHE);

    CachedActionResult cachedActionResult =
        remoteCache.downloadActionResult(
            action.getRemoteActionExecutionContext(),
            action.getActionKey(),
            /* inlineOutErr= */ false);

    if (cachedActionResult == null) {
      return null;
    }

    return RemoteActionResult.createFromCache(cachedActionResult);
  }

  private static Path toTmpDownloadPath(Path actualPath) {
    return checkNotNull(actualPath.getParentDirectory())
        .getRelative(actualPath.getBaseName() + ".tmp");
  }

  private ListenableFuture<FileMetadata> downloadFile(RemoteAction action, FileMetadata file) {
    checkNotNull(remoteCache, "remoteCache can't be null");

    try {
      ListenableFuture<Void> future =
          remoteCache.downloadFile(
              action.getRemoteActionExecutionContext(),
              remotePathResolver.localPathToOutputPath(file.path()),
              toTmpDownloadPath(file.path()),
              file.digest(),
              new RemoteCache.DownloadProgressReporter(
                  action.getSpawnExecutionContext()::report,
                  remotePathResolver.localPathToOutputPath(file.path()),
                  file.digest().getSizeBytes()));
      return transform(future, (d) -> file, directExecutor());
    } catch (IOException e) {
      return immediateFailedFuture(e);
    }
  }

  private void captureCorruptedOutputs(Exception e) {
    if (captureCorruptedOutputsDir != null) {
      if (e instanceof BulkTransferException) {
        for (Throwable suppressed : e.getSuppressed()) {
          if (suppressed instanceof OutputDigestMismatchException) {
            // Capture corrupted outputs
            try {
              String outputPath = ((OutputDigestMismatchException) suppressed).getOutputPath();
              Path localPath = ((OutputDigestMismatchException) suppressed).getLocalPath();
              Path dst = captureCorruptedOutputsDir.getRelative(outputPath);
              dst.createDirectoryAndParents();

              // Make sure dst is still under captureCorruptedOutputsDir, otherwise
              // IllegalArgumentException will be thrown.
              dst.relativeTo(captureCorruptedOutputsDir);

              FileSystemUtils.copyFile(localPath, dst);
            } catch (Exception ee) {
              ee.addSuppressed(ee);
            }
          }
        }
      }
    }
  }

  private void deletePartialDownloadedOutputs(
      RemoteActionResult result, FileOutErr tmpOutErr, Exception e) throws ExecException {
    try {
      // Delete any (partially) downloaded output files.
      for (OutputFile file : result.actionResult.getOutputFilesList()) {
        toTmpDownloadPath(remotePathResolver.outputPathToLocalPath(file.getPath())).delete();
      }

      for (OutputDirectory directory : result.actionResult.getOutputDirectoriesList()) {
        // Only delete the directories below the output directories because the output
        // directories will not be re-created
        remotePathResolver.outputPathToLocalPath(directory.getPath()).deleteTreesBelow();
      }

      tmpOutErr.clearOut();
      tmpOutErr.clearErr();
    } catch (IOException ioEx) {
      ioEx.addSuppressed(e);

      // If deleting of output files failed, we abort the build with a decent error message as
      // any subsequent local execution failure would likely be incomprehensible.
      ExecException execEx =
          new EnvironmentalExecException(
              ioEx,
              createFailureDetail(
                  "Failed to delete output files after incomplete download",
                  RemoteExecution.Code.INCOMPLETE_OUTPUT_DOWNLOAD_CLEANUP_FAILURE));
      execEx.addSuppressed(e);
      throw execEx;
    }
  }

  /**
   * Copies moves the downloaded outputs from their download location to their declared location.
   */
  private void moveOutputsToFinalLocation(List<ListenableFuture<FileMetadata>> downloads)
      throws IOException, InterruptedException {
    List<FileMetadata> finishedDownloads = new ArrayList<>(downloads.size());
    for (ListenableFuture<FileMetadata> finishedDownload : downloads) {
      FileMetadata outputFile = getFromFuture(finishedDownload);
      if (outputFile != null) {
        finishedDownloads.add(outputFile);
      }
    }
    /*
     * Sort the list lexicographically based on its temporary download path in order to avoid
     * filename clashes when moving the files:
     *
     * Consider an action that produces two outputs foo and foo.tmp. These outputs would initially
     * be downloaded to foo.tmp and foo.tmp.tmp. When renaming them to foo and foo.tmp we need to
     * ensure that rename(foo.tmp, foo) happens before rename(foo.tmp.tmp, foo.tmp). We ensure this
     * by doing the renames in lexicographical order of the download names.
     */
    Collections.sort(finishedDownloads, Comparator.comparing(f -> toTmpDownloadPath(f.path())));

    // Move the output files from their temporary name to the actual output file name. Executable
    // bit
    // is ignored since the file permission will be changed to 0555 after execution.
    for (FileMetadata outputFile : finishedDownloads) {
      FileSystemUtils.moveFile(toTmpDownloadPath(outputFile.path()), outputFile.path());
    }
  }

  private void createSymlinks(Iterable<SymlinkMetadata> symlinks) throws IOException {
    for (SymlinkMetadata symlink : symlinks) {
      if (symlink.target().isAbsolute()) {
        // We do not support absolute symlinks as outputs.
        throw new IOException(
            String.format(
                "Action output %s is a symbolic link to an absolute path %s. "
                    + "Symlinks to absolute paths in action outputs are not supported.",
                symlink.path(), symlink.target()));
      }
      Preconditions.checkNotNull(
              symlink.path().getParentDirectory(),
              "Failed creating directory and parents for %s",
              symlink.path())
          .createDirectoryAndParents();
      symlink.path().createSymbolicLink(symlink.target());
    }
  }

  private void injectRemoteArtifact(
      RemoteAction action, Artifact output, ActionResultMetadata metadata) throws IOException {
    RemoteActionExecutionContext context = action.getRemoteActionExecutionContext();
    MetadataInjector metadataInjector = action.getSpawnExecutionContext().getMetadataInjector();
    Path path = remotePathResolver.outputPathToLocalPath(output);
    if (output.isTreeArtifact()) {
      DirectoryMetadata directory = metadata.directory(path);
      if (directory == null) {
        // A declared output wasn't created. It might have been an optional output and if not
        // SkyFrame will make sure to fail.
        return;
      }
      if (!directory.symlinks().isEmpty()) {
        throw new IOException(
            "Symlinks in action outputs are not yet supported by "
                + "--experimental_remote_download_outputs=minimal");
      }
      SpecialArtifact parent = (SpecialArtifact) output;
      TreeArtifactValue.Builder tree = TreeArtifactValue.newBuilder(parent);
      for (FileMetadata file : directory.files()) {
        TreeFileArtifact child =
            TreeFileArtifact.createTreeOutput(parent, file.path().relativeTo(parent.getPath()));
        RemoteFileArtifactValue value =
            new RemoteFileArtifactValue(
                DigestUtil.toBinaryDigest(file.digest()),
                file.digest().getSizeBytes(),
                /*locationIndex=*/ 1,
                context.getRequestMetadata().getActionId());
        tree.putChild(child, value);
      }
      metadataInjector.injectTree(parent, tree.build());
    } else {
      FileMetadata outputMetadata = metadata.file(path);
      if (outputMetadata == null) {
        // A declared output wasn't created. It might have been an optional output and if not
        // SkyFrame will make sure to fail.
        return;
      }
      metadataInjector.injectFile(
          output,
          new RemoteFileArtifactValue(
              DigestUtil.toBinaryDigest(outputMetadata.digest()),
              outputMetadata.digest().getSizeBytes(),
              /*locationIndex=*/ 1,
              context.getRequestMetadata().getActionId()));
    }
  }

  /** In-memory representation of action result metadata. */
  static class ActionResultMetadata {

    static class SymlinkMetadata {
      private final Path path;
      private final PathFragment target;

      private SymlinkMetadata(Path path, PathFragment target) {
        this.path = path;
        this.target = target;
      }

      public Path path() {
        return path;
      }

      public PathFragment target() {
        return target;
      }
    }

    static class FileMetadata {
      private final Path path;
      private final Digest digest;
      private final boolean isExecutable;

      private FileMetadata(Path path, Digest digest, boolean isExecutable) {
        this.path = path;
        this.digest = digest;
        this.isExecutable = isExecutable;
      }

      public Path path() {
        return path;
      }

      public Digest digest() {
        return digest;
      }

      public boolean isExecutable() {
        return isExecutable;
      }
    }

    static class DirectoryMetadata {
      private final ImmutableList<FileMetadata> files;
      private final ImmutableList<SymlinkMetadata> symlinks;

      private DirectoryMetadata(
          ImmutableList<FileMetadata> files, ImmutableList<SymlinkMetadata> symlinks) {
        this.files = files;
        this.symlinks = symlinks;
      }

      public ImmutableList<FileMetadata> files() {
        return files;
      }

      public ImmutableList<SymlinkMetadata> symlinks() {
        return symlinks;
      }
    }

    private final ImmutableMap<Path, FileMetadata> files;
    private final ImmutableMap<Path, SymlinkMetadata> symlinks;
    private final ImmutableMap<Path, DirectoryMetadata> directories;

    private ActionResultMetadata(
        ImmutableMap<Path, FileMetadata> files,
        ImmutableMap<Path, SymlinkMetadata> symlinks,
        ImmutableMap<Path, DirectoryMetadata> directories) {
      this.files = files;
      this.symlinks = symlinks;
      this.directories = directories;
    }

    @Nullable
    public FileMetadata file(Path path) {
      return files.get(path);
    }

    @Nullable
    public DirectoryMetadata directory(Path path) {
      return directories.get(path);
    }

    public Collection<FileMetadata> files() {
      return files.values();
    }

    public ImmutableSet<Entry<Path, DirectoryMetadata>> directories() {
      return directories.entrySet();
    }

    public Collection<SymlinkMetadata> symlinks() {
      return symlinks.values();
    }
  }

  private DirectoryMetadata parseDirectory(
      Path parent, Directory dir, Map<Digest, Directory> childDirectoriesMap) {
    ImmutableList.Builder<FileMetadata> filesBuilder = ImmutableList.builder();
    for (FileNode file : dir.getFilesList()) {
      filesBuilder.add(
          new FileMetadata(
              parent.getRelative(file.getName()), file.getDigest(), file.getIsExecutable()));
    }

    ImmutableList.Builder<SymlinkMetadata> symlinksBuilder = ImmutableList.builder();
    for (SymlinkNode symlink : dir.getSymlinksList()) {
      symlinksBuilder.add(
          new SymlinkMetadata(
              parent.getRelative(symlink.getName()), PathFragment.create(symlink.getTarget())));
    }

    for (DirectoryNode directoryNode : dir.getDirectoriesList()) {
      Path childPath = parent.getRelative(directoryNode.getName());
      Directory childDir =
          Preconditions.checkNotNull(childDirectoriesMap.get(directoryNode.getDigest()));
      DirectoryMetadata childMetadata = parseDirectory(childPath, childDir, childDirectoriesMap);
      filesBuilder.addAll(childMetadata.files());
      symlinksBuilder.addAll(childMetadata.symlinks());
    }

    return new DirectoryMetadata(filesBuilder.build(), symlinksBuilder.build());
  }

  ActionResultMetadata parseActionResultMetadata(RemoteAction action, RemoteActionResult result)
      throws IOException, InterruptedException {
    checkNotNull(remoteCache, "remoteCache can't be null");

    Map<Path, ListenableFuture<Tree>> dirMetadataDownloads =
        Maps.newHashMapWithExpectedSize(result.getOutputDirectoriesCount());
    for (OutputDirectory dir : result.getOutputDirectories()) {
      dirMetadataDownloads.put(
          remotePathResolver.outputPathToLocalPath(encodeBytestringUtf8(dir.getPath())),
          Futures.transformAsync(
              remoteCache.downloadBlob(
                  action.getRemoteActionExecutionContext(), dir.getTreeDigest()),
              (treeBytes) ->
                  immediateFuture(Tree.parseFrom(treeBytes, ExtensionRegistry.getEmptyRegistry())),
              directExecutor()));
    }

    waitForBulkTransfer(dirMetadataDownloads.values(), /* cancelRemainingOnInterrupt=*/ true);

    ImmutableMap.Builder<Path, DirectoryMetadata> directories = ImmutableMap.builder();
    for (Map.Entry<Path, ListenableFuture<Tree>> metadataDownload :
        dirMetadataDownloads.entrySet()) {
      Path path = metadataDownload.getKey();
      Tree directoryTree = getFromFuture(metadataDownload.getValue());
      Map<Digest, Directory> childrenMap = new HashMap<>();
      for (Directory childDir : directoryTree.getChildrenList()) {
        childrenMap.put(digestUtil.compute(childDir), childDir);
      }

      directories.put(path, parseDirectory(path, directoryTree.getRoot(), childrenMap));
    }

    ImmutableMap.Builder<Path, FileMetadata> files = ImmutableMap.builder();
    for (OutputFile outputFile : result.getOutputFiles()) {
      Path localPath =
          remotePathResolver.outputPathToLocalPath(encodeBytestringUtf8(outputFile.getPath()));
      files.put(
          localPath,
          new FileMetadata(localPath, outputFile.getDigest(), outputFile.getIsExecutable()));
    }

    ImmutableMap.Builder<Path, SymlinkMetadata> symlinks = ImmutableMap.builder();
    Iterable<OutputSymlink> outputSymlinks =
        Iterables.concat(result.getOutputFileSymlinks(), result.getOutputDirectorySymlinks());
    for (OutputSymlink symlink : outputSymlinks) {
      Path localPath =
          remotePathResolver.outputPathToLocalPath(encodeBytestringUtf8(symlink.getPath()));
      symlinks.put(
          localPath, new SymlinkMetadata(localPath, PathFragment.create(symlink.getTarget())));
    }

    return new ActionResultMetadata(
        files.buildOrThrow(), symlinks.buildOrThrow(), directories.buildOrThrow());
  }

  /**
   * Download the output files and directory trees of a remotely executed action, as well stdin /
   * stdout to the local machine.
   *
   * <p>If minimal download mode is determined, this method avoids downloading the majority of
   * action outputs by inject their metadata. In this case, this method only downloads output
   * directory metadata, stdout and stderr as well as the contents of {@code
   * ExecutionRequirements.REMOTE_EXECUTION_INLINE_OUTPUTS} if specified.
   *
   * <p>In case of failure, this method deletes any output files it might have already created.
   */
  @Nullable
  public InMemoryOutput downloadOutputs(RemoteAction action, RemoteActionResult result)
      throws InterruptedException, IOException, ExecException {
    checkState(!shutdown.get(), "shutdown");
    checkNotNull(remoteCache, "remoteCache can't be null");

    action.getRemoteActionExecutionContext().setStep(Step.DOWNLOAD_OUTPUTS);

    ActionResultMetadata metadata;
    try (SilentCloseable c = Profiler.instance().profile("Remote.parseActionResultMetadata")) {
      metadata = parseActionResultMetadata(action, result);
    }

    if (result.success()) {
      // Check that all mandatory outputs are created.
      for (ActionInput output : action.getSpawn().getOutputFiles()) {
        if (action.getSpawn().isMandatoryOutput(output)) {
          // In the past, remote execution did not create output directories if the action didn't do
          // this explicitly. This check only remains so that old remote cache entries that do not
          // include empty output directories remain valid.
          if (output instanceof Artifact && ((Artifact) output).isTreeArtifact()) {
            continue;
          }

          Path localPath = execRoot.getRelative(output.getExecPath());
          if (!metadata.files.containsKey(localPath)
              && !metadata.directories.containsKey(localPath)
              && !metadata.symlinks.containsKey(localPath)) {
            throw new IOException(
                "Invalid action cache entry "
                    + action.getActionKey().getDigest().getHash()
                    + ": expected output "
                    + prettyPrint(output)
                    + " does not exist.");
          }
        }
      }

      // When downloading outputs from just remotely executed action, the action result comes from
      // Execution response which means, if disk cache is enabled, action result hasn't been
      // uploaded to it. Upload action result to disk cache here so next build could hit it.
      if (useDiskCache(remoteOptions)
          && action.getRemoteActionExecutionContext().getExecuteResponse() != null) {
        getFromFuture(
            remoteCache.uploadActionResult(
                action.getRemoteActionExecutionContext(),
                action.getActionKey(),
                result.actionResult));
      }
    }

    FileOutErr outErr = action.getSpawnExecutionContext().getFileOutErr();

    ImmutableList.Builder<ListenableFuture<FileMetadata>> downloadsBuilder =
        ImmutableList.builder();
    RemoteOutputsMode remoteOutputsMode = remoteOptions.remoteOutputsMode;
    boolean downloadOutputs =
        shouldDownloadAllSpawnOutputs(
            remoteOutputsMode,
            /* exitCode = */ result.getExitCode(),
            hasFilesToDownload(action.getSpawn().getOutputFiles(), filesToDownload));

    if (downloadOutputs) {
      HashSet<PathFragment> queuedFilePaths = new HashSet<>();

      for (FileMetadata file : metadata.files()) {
        PathFragment filePath = file.path().asFragment();
        if (queuedFilePaths.add(filePath)) {
          downloadsBuilder.add(downloadFile(action, file));
        }
      }

      for (Map.Entry<Path, DirectoryMetadata> entry : metadata.directories()) {
        for (FileMetadata file : entry.getValue().files()) {
          PathFragment filePath = file.path().asFragment();
          if (queuedFilePaths.add(filePath)) {
            downloadsBuilder.add(downloadFile(action, file));
          }
        }
      }
    } else {
      checkState(
          result.getExitCode() == 0,
          "injecting remote metadata is only supported for successful actions (exit code 0).");

      if (!metadata.symlinks().isEmpty()) {
        throw new IOException(
            "Symlinks in action outputs are not yet supported by "
                + "--experimental_remote_download_outputs=minimal");
      }
    }

    FileOutErr tmpOutErr = outErr.childOutErr();
    List<ListenableFuture<Void>> outErrDownloads =
        remoteCache.downloadOutErr(
            action.getRemoteActionExecutionContext(), result.actionResult, tmpOutErr);
    for (ListenableFuture<Void> future : outErrDownloads) {
      downloadsBuilder.add(transform(future, (v) -> null, directExecutor()));
    }

    ImmutableList<ListenableFuture<FileMetadata>> downloads = downloadsBuilder.build();
    try (SilentCloseable c = Profiler.instance().profile("Remote.download")) {
      waitForBulkTransfer(downloads, /* cancelRemainingOnInterrupt= */ true);
    } catch (Exception e) {
      captureCorruptedOutputs(e);
      deletePartialDownloadedOutputs(result, tmpOutErr, e);
      throw e;
    }

    FileOutErr.dump(tmpOutErr, outErr);

    // Ensure that we are the only ones writing to the output files when using the dynamic spawn
    // strategy.
    action
        .getSpawnExecutionContext()
        .lockOutputFiles(result.getExitCode(), result.getMessage(), tmpOutErr);
    // Will these be properly garbage-collected if the above throws an exception?
    tmpOutErr.clearOut();
    tmpOutErr.clearErr();

    if (downloadOutputs) {
      moveOutputsToFinalLocation(downloads);

      List<SymlinkMetadata> symlinksInDirectories = new ArrayList<>();
      for (Entry<Path, DirectoryMetadata> entry : metadata.directories()) {
        // For empty output directories, create the empty directory itself
        entry.getKey().createDirectoryAndParents();
        symlinksInDirectories.addAll(entry.getValue().symlinks());
      }

      Iterable<SymlinkMetadata> symlinks =
          Iterables.concat(metadata.symlinks(), symlinksInDirectories);

      // Create the symbolic links after all downloads are finished, because dangling symlinks
      // might not be supported on all platforms
      createSymlinks(symlinks);
    } else {
      ActionInput inMemoryOutput = null;
      Digest inMemoryOutputDigest = null;
      PathFragment inMemoryOutputPath = getInMemoryOutputPath(action.getSpawn());

      for (ActionInput output : action.getSpawn().getOutputFiles()) {
        if (inMemoryOutputPath != null && output.getExecPath().equals(inMemoryOutputPath)) {
          Path localPath = remotePathResolver.outputPathToLocalPath(output);
          FileMetadata m = metadata.file(localPath);
          if (m == null) {
            // A declared output wasn't created. Ignore it here. SkyFrame will fail if not all
            // outputs were created.
            continue;
          }
          inMemoryOutputDigest = m.digest();
          inMemoryOutput = output;
        }
        if (output instanceof Artifact) {
          injectRemoteArtifact(action, (Artifact) output, metadata);
        }
      }

      try (SilentCloseable c = Profiler.instance().profile("Remote.downloadInMemoryOutput")) {
        if (inMemoryOutput != null) {
          ListenableFuture<byte[]> inMemoryOutputDownload =
              remoteCache.downloadBlob(
                  action.getRemoteActionExecutionContext(), inMemoryOutputDigest);
          waitForBulkTransfer(
              ImmutableList.of(inMemoryOutputDownload), /* cancelRemainingOnInterrupt=*/ true);
          byte[] data = getFromFuture(inMemoryOutputDownload);
          return new InMemoryOutput(inMemoryOutput, ByteString.copyFrom(data));
        }
      }
    }

    return null;
  }

  private static String prettyPrint(ActionInput actionInput) {
    if (actionInput instanceof Artifact) {
      return ((Artifact) actionInput).prettyPrint();
    } else {
      return actionInput.getExecPathString();
    }
  }

  private Single<UploadManifest> buildUploadManifestAsync(
      RemoteAction action, SpawnResult spawnResult) {
    return Single.fromCallable(
        () -> {
          ImmutableList.Builder<Path> outputFiles = ImmutableList.builder();
          // Check that all mandatory outputs are created.
          for (ActionInput outputFile : action.getSpawn().getOutputFiles()) {
            Path localPath = execRoot.getRelative(outputFile.getExecPath());
            if (action.getSpawn().isMandatoryOutput(outputFile) && !localPath.exists()) {
              throw new IOException(
                  "Expected output " + prettyPrint(outputFile) + " was not created locally.");
            }
            outputFiles.add(localPath);
          }

          return UploadManifest.create(
              remoteOptions,
              digestUtil,
              remotePathResolver,
              action.getActionKey(),
              action.getAction(),
              action.getCommand(),
              outputFiles.build(),
              action.getSpawnExecutionContext().getFileOutErr(),
              spawnResult.exitCode(),
              spawnResult.getStartTime(),
              spawnResult.getWallTime());
        });
  }

  @VisibleForTesting
  UploadManifest buildUploadManifest(RemoteAction action, SpawnResult spawnResult)
      throws IOException, ExecException, InterruptedException {
    try {
      return buildUploadManifestAsync(action, spawnResult).blockingGet();
    } catch (RuntimeException e) {
      Throwable cause = e.getCause();
      if (cause != null) {
        Throwables.throwIfInstanceOf(cause, IOException.class);
        Throwables.throwIfInstanceOf(cause, ExecException.class);
        Throwables.throwIfInstanceOf(cause, InterruptedException.class);
      }
      throw e;
    }
  }

  /** Upload outputs of a remote action which was executed locally to remote cache. */
  public void uploadOutputs(RemoteAction action, SpawnResult spawnResult)
      throws InterruptedException, ExecException {
    checkState(!shutdown.get(), "shutdown");
    checkState(shouldUploadLocalResults(action.getSpawn()), "spawn shouldn't upload local result");
    checkState(
        SpawnResult.Status.SUCCESS.equals(spawnResult.status()) && spawnResult.exitCode() == 0,
        "shouldn't upload outputs of failed local action");

    if (remoteOptions.remoteCacheAsync) {
      Single.using(
              remoteCache::retain,
              remoteCache ->
                  buildUploadManifestAsync(action, spawnResult)
                      .flatMap(
                          manifest ->
                              manifest.uploadAsync(
                                  action.getRemoteActionExecutionContext(), remoteCache, reporter)),
              RemoteCache::release)
          .subscribeOn(scheduler)
          .subscribe(
              new SingleObserver<ActionResult>() {
                @Override
                public void onSubscribe(@NonNull Disposable d) {
                  backgroundTaskPhaser.register();
                }

                @Override
                public void onSuccess(@NonNull ActionResult actionResult) {
                  backgroundTaskPhaser.arriveAndDeregister();
                }

                @Override
                public void onError(@NonNull Throwable e) {
                  backgroundTaskPhaser.arriveAndDeregister();
                  reportUploadError(e);
                }
              });
    } else {
      try (SilentCloseable c =
          Profiler.instance().profile(ProfilerTask.UPLOAD_TIME, "upload outputs")) {
        UploadManifest manifest = buildUploadManifest(action, spawnResult);
        manifest.upload(action.getRemoteActionExecutionContext(), remoteCache, reporter);
      } catch (IOException e) {
        reportUploadError(e);
      }
    }
  }

  private void reportUploadError(Throwable error) {
    if (buildInterrupted.get()) {
      // If build interrupted, ignores all the errors
      return;
    }

    String errorMessage = "Remote Cache: " + grpcAwareErrorMessage(error, verboseFailures);

    report(Event.warn(errorMessage));
  }

  /**
   * Upload inputs of a remote action to remote cache if they are not presented already.
   *
   * <p>Must be called before calling {@link #executeRemotely}.
   */
  public void uploadInputsIfNotPresent(RemoteAction action, boolean force)
      throws IOException, InterruptedException {
    checkState(!shutdown.get(), "shutdown");
    checkState(mayBeExecutedRemotely(action.getSpawn()), "spawn can't be executed remotely");

    action.getRemoteActionExecutionContext().setStep(Step.UPLOAD_INPUTS);

    RemoteExecutionCache remoteExecutionCache = (RemoteExecutionCache) remoteCache;
    // Upload the command and all the inputs into the remote cache.
    Map<Digest, Message> additionalInputs = Maps.newHashMapWithExpectedSize(2);
    additionalInputs.put(action.getActionKey().getDigest(), action.getAction());
    additionalInputs.put(action.getCommandHash(), action.getCommand());
    remoteExecutionCache.ensureInputsPresent(
        action.getRemoteActionExecutionContext(), action.getMerkleTree(), additionalInputs, force);
  }

  /**
   * Executes the remote action remotely and returns the result.
   *
   * @param acceptCachedResult tells remote execution server whether it should used cached result.
   * @param observer receives status updates during the execution.
   */
  public RemoteActionResult executeRemotely(
      RemoteAction action, boolean acceptCachedResult, OperationObserver observer)
      throws IOException, InterruptedException {
    checkState(!shutdown.get(), "shutdown");
    checkState(mayBeExecutedRemotely(action.getSpawn()), "spawn can't be executed remotely");

    action.getRemoteActionExecutionContext().setStep(Step.EXECUTE_REMOTELY);

    ExecuteRequest.Builder requestBuilder =
        ExecuteRequest.newBuilder()
            .setInstanceName(remoteOptions.remoteInstanceName)
            .setActionDigest(action.getActionKey().getDigest())
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
        remoteExecutor.executeRemotely(action.getRemoteActionExecutionContext(), request, observer);

    action.getRemoteActionExecutionContext().setExecuteResponse(reply);

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
    checkState(!shutdown.get(), "shutdown");
    checkNotNull(remoteCache, "remoteCache can't be null");
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
                  action.getRemoteActionExecutionContext(),
                  serverLogs.lastLogPath,
                  e.getValue().getDigest()));
        }
      }
    }

    return serverLogs;
  }

  @Subscribe
  public void onBuildInterrupted(BuildInterruptedEvent event) {
    buildInterrupted.set(true);
  }

  /**
   * Shuts the service down. Wait for active network I/O to finish but new requests are rejected.
   */
  public void shutdown() {
    if (!shutdown.compareAndSet(false, true)) {
      return;
    }

    if (buildInterrupted.get()) {
      Thread.currentThread().interrupt();
    }

    if (remoteCache != null) {
      remoteCache.release();

      try {
        backgroundTaskPhaser.awaitAdvanceInterruptibly(backgroundTaskPhaser.arrive());
      } catch (InterruptedException e) {
        buildInterrupted.set(true);
        remoteCache.shutdownNow();
        Thread.currentThread().interrupt();
      }
    }

    if (remoteExecutor != null) {
      remoteExecutor.close();
    }
  }

  void report(Event evt) {

    synchronized (this) {
      if (reportedErrors.contains(evt.getMessage())) {
        return;
      }
      reportedErrors.add(evt.getMessage());
      reporter.handle(evt);
    }
  }
}

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
import static com.google.devtools.build.lib.remote.RemoteCache.waitForBulkTransfer;
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
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.ForbiddenActionInputException;
import com.google.devtools.build.lib.actions.RemoteUploadEvent;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.actions.cache.MetadataInjector;
import com.google.devtools.build.lib.analysis.platform.PlatformUtils;
import com.google.devtools.build.lib.buildtool.buildevent.BuildInterruptedEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.RemoteExecutionService.ActionResultMetadata.DirectoryMetadata;
import com.google.devtools.build.lib.remote.RemoteExecutionService.ActionResultMetadata.FileMetadata;
import com.google.devtools.build.lib.remote.RemoteExecutionService.ActionResultMetadata.SymlinkMetadata;
import com.google.devtools.build.lib.remote.common.NetworkTime;
import com.google.devtools.build.lib.remote.common.OperationObserver;
import com.google.devtools.build.lib.remote.common.OutputDigestMismatchException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.common.RemoteExecutionClient;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.RxFutures;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.remote.util.Utils.InMemoryOutput;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.RemoteExecution;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.protobuf.ByteString;
import com.google.protobuf.ExtensionRegistry;
import com.google.protobuf.Message;
import io.grpc.Status.Code;
import io.reactivex.rxjava3.annotations.NonNull;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.CompletableObserver;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.disposables.Disposable;
import io.reactivex.rxjava3.schedulers.Schedulers;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.SortedMap;
import java.util.TreeSet;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/**
 * A layer between spawn execution and remote execution exposing primitive operations for remote
 * cache and execution with spawn specific types.
 */
public class RemoteExecutionService {
  private final AtomicBoolean shutdown = new AtomicBoolean(false);
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

  private boolean remoteCacheInterrupted = false;

  public RemoteExecutionService(
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

    ImmutableSet.Builder<PathFragment> filesToDownloadBuilder = ImmutableSet.builder();
    for (ActionInput actionInput : filesToDownload) {
      filesToDownloadBuilder.add(actionInput.getExecPath());
    }
    this.filesToDownload = filesToDownloadBuilder.build();
    this.captureCorruptedOutputsDir = captureCorruptedOutputsDir;
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

    public RemoteActionExecutionContext getRemoteActionExecutionContext() {
      return remoteActionExecutionContext;
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

  /** Creates a new {@link RemoteAction} instance from spawn. */
  public RemoteAction buildRemoteAction(Spawn spawn, SpawnExecutionContext context)
      throws IOException, UserExecException, ForbiddenActionInputException {
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
        RemoteActionExecutionContext.createForSpawn(spawn, metadata);

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
    checkState(!shutdown.get(), "shutdown");
    checkState(shouldAcceptCachedResult(action.spawn), "spawn doesn't accept cached result");

    ActionResult actionResult =
        remoteCache.downloadActionResult(
            action.remoteActionExecutionContext, action.actionKey, /* inlineOutErr= */ false);

    if (actionResult == null) {
      return null;
    }

    return RemoteActionResult.createFromCache(actionResult);
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
              action.remoteActionExecutionContext,
              remotePathResolver.localPathToOutputPath(file.path()),
              toTmpDownloadPath(file.path()),
              file.digest(),
              new RemoteCache.DownloadProgressReporter(
                  action.spawnExecutionContext::report,
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

    // Move the output files from their temporary name to the actual output file name.
    for (FileMetadata outputFile : finishedDownloads) {
      FileSystemUtils.moveFile(toTmpDownloadPath(outputFile.path()), outputFile.path());
      outputFile.path().setExecutable(outputFile.isExecutable());
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
    RemoteActionExecutionContext context = action.remoteActionExecutionContext;
    MetadataInjector metadataInjector = action.spawnExecutionContext.getMetadataInjector();
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
          remotePathResolver.outputPathToLocalPath(dir.getPath()),
          Futures.transformAsync(
              remoteCache.downloadBlob(action.remoteActionExecutionContext, dir.getTreeDigest()),
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
      Path localPath = remotePathResolver.outputPathToLocalPath(outputFile.getPath());
      files.put(
          localPath,
          new FileMetadata(localPath, outputFile.getDigest(), outputFile.getIsExecutable()));
    }

    ImmutableMap.Builder<Path, SymlinkMetadata> symlinks = ImmutableMap.builder();
    Iterable<OutputSymlink> outputSymlinks =
        Iterables.concat(result.getOutputFileSymlinks(), result.getOutputDirectorySymlinks());
    for (OutputSymlink symlink : outputSymlinks) {
      Path localPath = remotePathResolver.outputPathToLocalPath(symlink.getPath());
      symlinks.put(
          localPath, new SymlinkMetadata(localPath, PathFragment.create(symlink.getTarget())));
    }

    return new ActionResultMetadata(files.build(), symlinks.build(), directories.build());
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

    ActionResultMetadata metadata;
    try (SilentCloseable c = Profiler.instance().profile("Remote.parseActionResultMetadata")) {
      metadata = parseActionResultMetadata(action, result);
    }

    FileOutErr outErr = action.spawnExecutionContext.getFileOutErr();

    ImmutableList.Builder<ListenableFuture<FileMetadata>> downloadsBuilder =
        ImmutableList.builder();
    RemoteOutputsMode remoteOutputsMode = remoteOptions.remoteOutputsMode;
    boolean downloadOutputs =
        shouldDownloadAllSpawnOutputs(
            remoteOutputsMode,
            /* exitCode = */ result.getExitCode(),
            hasFilesToDownload(action.spawn.getOutputFiles(), filesToDownload));

    if (downloadOutputs) {
      for (FileMetadata file : metadata.files()) {
        downloadsBuilder.add(downloadFile(action, file));
      }

      for (Map.Entry<Path, DirectoryMetadata> entry : metadata.directories()) {
        for (FileMetadata file : entry.getValue().files()) {
          downloadsBuilder.add(downloadFile(action, file));
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
            action.remoteActionExecutionContext, result.actionResult, tmpOutErr);
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
    tmpOutErr.clearOut();
    tmpOutErr.clearErr();

    // Ensure that we are the only ones writing to the output files when using the dynamic spawn
    // strategy.
    action.spawnExecutionContext.lockOutputFiles();

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
      PathFragment inMemoryOutputPath = getInMemoryOutputPath(action.spawn);

      for (ActionInput output : action.spawn.getOutputFiles()) {
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
              remoteCache.downloadBlob(action.remoteActionExecutionContext, inMemoryOutputDigest);
          waitForBulkTransfer(
              ImmutableList.of(inMemoryOutputDownload), /* cancelRemainingOnInterrupt=*/ true);
          byte[] data = getFromFuture(inMemoryOutputDownload);
          return new InMemoryOutput(inMemoryOutput, ByteString.copyFrom(data));
        }
      }
    }

    return null;
  }

  private static FailureDetail createFailureDetail(
      String message, RemoteExecution.Code detailedCode) {
    return FailureDetail.newBuilder()
        .setMessage(message)
        .setRemoteExecution(RemoteExecution.newBuilder().setCode(detailedCode))
        .build();
  }

  /** UploadManifest adds output metadata to a {@link ActionResult}. */
  static class UploadManifest {
    private final DigestUtil digestUtil;
    private final RemotePathResolver remotePathResolver;
    private final ActionResult.Builder result;
    private final boolean allowSymlinks;
    private final boolean uploadSymlinks;
    private final Map<Digest, Path> digestToFile = new HashMap<>();
    private final Map<Digest, ByteString> digestToBlobs = new HashMap<>();
    private Digest stderrDigest;
    private Digest stdoutDigest;

    /**
     * Create an UploadManifest from an ActionResult builder and an exec root. The ActionResult
     * builder is populated through a call to {@link #addFile(Digest, Path)}.
     */
    public UploadManifest(
        DigestUtil digestUtil,
        RemotePathResolver remotePathResolver,
        ActionResult.Builder result,
        boolean uploadSymlinks,
        boolean allowSymlinks) {
      this.digestUtil = digestUtil;
      this.remotePathResolver = remotePathResolver;
      this.result = result;
      this.uploadSymlinks = uploadSymlinks;
      this.allowSymlinks = allowSymlinks;
    }

    public void setStdoutStderr(FileOutErr outErr) throws IOException {
      if (outErr.getErrorPath().exists()) {
        stderrDigest = digestUtil.compute(outErr.getErrorPath());
        digestToFile.put(stderrDigest, outErr.getErrorPath());
      }
      if (outErr.getOutputPath().exists()) {
        stdoutDigest = digestUtil.compute(outErr.getOutputPath());
        digestToFile.put(stdoutDigest, outErr.getOutputPath());
      }
    }

    /**
     * Add a collection of files or directories to the UploadManifest. Adding a directory has the
     * effect of 1) uploading a {@link Tree} protobuf message from which the whole structure of the
     * directory, including the descendants, can be reconstructed and 2) uploading all the
     * non-directory descendant files.
     */
    public void addFiles(Collection<Path> files) throws ExecException, IOException {
      for (Path file : files) {
        // TODO(ulfjack): Maybe pass in a SpawnResult here, add a list of output files to that, and
        // rely on the local spawn runner to stat the files, instead of statting here.
        FileStatus stat = file.statIfFound(Symlinks.NOFOLLOW);
        // TODO(#6547): handle the case where the parent directory of the output file is an
        // output symlink.
        if (stat == null) {
          // We ignore requested results that have not been generated by the action.
          continue;
        }
        if (stat.isDirectory()) {
          addDirectory(file);
        } else if (stat.isFile() && !stat.isSpecialFile()) {
          Digest digest = digestUtil.compute(file, stat.getSize());
          addFile(digest, file);
        } else if (stat.isSymbolicLink() && allowSymlinks) {
          PathFragment target = file.readSymbolicLink();
          // Need to resolve the symbolic link to know what to add, file or directory.
          FileStatus statFollow = file.statIfFound(Symlinks.FOLLOW);
          if (statFollow == null) {
            throw new IOException(
                String.format("Action output %s is a dangling symbolic link to %s ", file, target));
          }
          if (statFollow.isSpecialFile()) {
            illegalOutput(file);
          }
          Preconditions.checkState(
              statFollow.isFile() || statFollow.isDirectory(), "Unknown stat type for %s", file);
          if (uploadSymlinks && !target.isAbsolute()) {
            if (statFollow.isFile()) {
              addFileSymbolicLink(file, target);
            } else {
              addDirectorySymbolicLink(file, target);
            }
          } else {
            if (statFollow.isFile()) {
              addFile(digestUtil.compute(file), file);
            } else {
              addDirectory(file);
            }
          }
        } else {
          illegalOutput(file);
        }
      }
    }

    /**
     * Adds an action and command protos to upload. They need to be uploaded as part of the action
     * result.
     */
    public void addAction(RemoteCacheClient.ActionKey actionKey, Action action, Command command) {
      digestToBlobs.put(actionKey.getDigest(), action.toByteString());
      digestToBlobs.put(action.getCommandDigest(), command.toByteString());
    }

    /** Map of digests to file paths to upload. */
    public Map<Digest, Path> getDigestToFile() {
      return digestToFile;
    }

    /**
     * Map of digests to chunkers to upload. When the file is a regular, non-directory file it is
     * transmitted through {@link #getDigestToFile()}. When it is a directory, it is transmitted as
     * a {@link Tree} protobuf message through {@link #getDigestToBlobs()}.
     */
    public Map<Digest, ByteString> getDigestToBlobs() {
      return digestToBlobs;
    }

    @Nullable
    public Digest getStdoutDigest() {
      return stdoutDigest;
    }

    @Nullable
    public Digest getStderrDigest() {
      return stderrDigest;
    }

    private void addFileSymbolicLink(Path file, PathFragment target) throws IOException {
      result
          .addOutputFileSymlinksBuilder()
          .setPath(remotePathResolver.localPathToOutputPath(file))
          .setTarget(target.toString());
    }

    private void addDirectorySymbolicLink(Path file, PathFragment target) throws IOException {
      result
          .addOutputDirectorySymlinksBuilder()
          .setPath(remotePathResolver.localPathToOutputPath(file))
          .setTarget(target.toString());
    }

    private void addFile(Digest digest, Path file) throws IOException {
      result
          .addOutputFilesBuilder()
          .setPath(remotePathResolver.localPathToOutputPath(file))
          .setDigest(digest)
          .setIsExecutable(file.isExecutable());

      digestToFile.put(digest, file);
    }

    private void addDirectory(Path dir) throws ExecException, IOException {
      Tree.Builder tree = Tree.newBuilder();
      Directory root = computeDirectory(dir, tree);
      tree.setRoot(root);

      ByteString data = tree.build().toByteString();
      Digest digest = digestUtil.compute(data.toByteArray());

      if (result != null) {
        result
            .addOutputDirectoriesBuilder()
            .setPath(remotePathResolver.localPathToOutputPath(dir))
            .setTreeDigest(digest);
      }

      digestToBlobs.put(digest, data);
    }

    private Directory computeDirectory(Path path, Tree.Builder tree)
        throws ExecException, IOException {
      Directory.Builder b = Directory.newBuilder();

      List<Dirent> sortedDirent = new ArrayList<>(path.readdir(Symlinks.NOFOLLOW));
      sortedDirent.sort(Comparator.comparing(Dirent::getName));

      for (Dirent dirent : sortedDirent) {
        String name = dirent.getName();
        Path child = path.getRelative(name);
        if (dirent.getType() == Dirent.Type.DIRECTORY) {
          Directory dir = computeDirectory(child, tree);
          b.addDirectoriesBuilder().setName(name).setDigest(digestUtil.compute(dir));
          tree.addChildren(dir);
        } else if (dirent.getType() == Dirent.Type.SYMLINK && allowSymlinks) {
          PathFragment target = child.readSymbolicLink();
          if (uploadSymlinks && !target.isAbsolute()) {
            // Whether it is dangling or not, we're passing it on.
            b.addSymlinksBuilder().setName(name).setTarget(target.toString());
            continue;
          }
          // Need to resolve the symbolic link now to know whether to upload a file or a directory.
          FileStatus statFollow = child.statIfFound(Symlinks.FOLLOW);
          if (statFollow == null) {
            throw new IOException(
                String.format(
                    "Action output %s is a dangling symbolic link to %s ", child, target));
          }
          if (statFollow.isFile() && !statFollow.isSpecialFile()) {
            Digest digest = digestUtil.compute(child);
            b.addFilesBuilder()
                .setName(name)
                .setDigest(digest)
                .setIsExecutable(child.isExecutable());
            digestToFile.put(digest, child);
          } else if (statFollow.isDirectory()) {
            Directory dir = computeDirectory(child, tree);
            b.addDirectoriesBuilder().setName(name).setDigest(digestUtil.compute(dir));
            tree.addChildren(dir);
          } else {
            illegalOutput(child);
          }
        } else if (dirent.getType() == Dirent.Type.FILE) {
          Digest digest = digestUtil.compute(child);
          b.addFilesBuilder().setName(name).setDigest(digest).setIsExecutable(child.isExecutable());
          digestToFile.put(digest, child);
        } else {
          illegalOutput(child);
        }
      }

      return b.build();
    }

    private void illegalOutput(Path what) throws ExecException {
      String kind = what.isSymbolicLink() ? "symbolic link" : "special file";
      String message =
          String.format(
              "Output %s is a %s. Only regular files and directories may be "
                  + "uploaded to a remote cache. "
                  + "Change the file type or use --remote_allow_symlink_upload.",
              remotePathResolver.localPathToOutputPath(what), kind);
      throw new UserExecException(
          createFailureDetail(message, RemoteExecution.Code.ILLEGAL_OUTPUT));
    }
  }

  private Completable uploadOutputs(
      RemoteCache remoteCache, RemoteAction action, UploadManifest manifest) {
    RemoteActionExecutionContext context = action.remoteActionExecutionContext;
    Map<Digest, Path> digestToFile = manifest.getDigestToFile();
    Map<Digest, ByteString> digestToBlobs = manifest.getDigestToBlobs();
    Collection<Digest> digests = new ArrayList<>();
    digests.addAll(digestToFile.keySet());
    digests.addAll(digestToBlobs.keySet());
    return RxFutures.toSingle(
            () -> remoteCache.findMissingDigests(context, digests), directExecutor())
        .flatMapPublisher(Flowable::fromIterable)
        .flatMapCompletable(
            digest -> {
              Path file = digestToFile.get(digest);
              Completable completable;

              String resourceId = "cas/" + digest.getHash();
              StringBuilder progressBuilder = new StringBuilder("Uploading ");

              if (file != null) {
                completable = RxFutures.toCompletable(
                    () -> remoteCache.uploadFile(context, digest, file), directExecutor());
                progressBuilder.append("file ").append(file.relativeTo(execRoot).getPathString());
              } else {
                ByteString blob = digestToBlobs.get(digest);
                if (blob == null) {
                  String message = "FindMissingBlobs call returned an unknown digest: " + digest;
                  return Completable.error(new IOException(message));
                }
                completable = RxFutures.toCompletable(
                    () -> remoteCache.uploadBlob(context, digest, blob), directExecutor());
                progressBuilder.append("blob ").append(digest.getHash());
              }

              String progress = progressBuilder.toString();

              return completable
                  .doOnSubscribe(ignored -> reporter.post(RemoteUploadEvent.create(resourceId, progress, false)))
                  .doFinally(() -> reporter.post(RemoteUploadEvent.create(resourceId, progress, true)));
            });
  }

  private Completable uploadActionResult(RemoteCache remoteCache, RemoteAction action, ActionResult result) {
    if (action.action.getDoNotCache()) {
      return Completable.complete();
    }

    String resourceId = "ac/" + action.actionKey.getDigest().getHash();
    StringBuilder progressBuilder = new StringBuilder("Uploading action result for action ");
    if (action.spawn.getResourceOwner() != null) {
      progressBuilder.append(action.spawn.getResourceOwner().describe());
    } else {
      progressBuilder.append(action.actionKey.getDigest());
    }
    String progress = progressBuilder.toString();

    return RxFutures.toCompletable(
            () ->
                remoteCache.uploadActionResult(
                    action.remoteActionExecutionContext, action.actionKey, result),
            directExecutor())
            .doOnSubscribe(
                ignored -> reporter.post(RemoteUploadEvent.create(resourceId, progress, false)))
            .doFinally(() -> reporter.post(RemoteUploadEvent.create(resourceId, progress, true)));
  }

  /** Upload outputs of a remote action which was executed locally to remote cache. */
  public void uploadOutputs(RemoteAction action)
      throws IOException, ExecException, InterruptedException {
    checkState(!shutdown.get(), "shutdown");
    checkState(shouldUploadLocalResults(action.spawn), "spawn shouldn't upload local result");

    CountDownLatch uploadStartedLatch = new CountDownLatch(1);

    Completable.using(
            remoteCache::retain,
            remoteCache -> {
              uploadStartedLatch.countDown();

              Collection<Path> outputFiles =
                  action.spawn.getOutputFiles().stream()
                      .map((inp) -> execRoot.getRelative(inp.getExecPath()))
                      .collect(ImmutableList.toImmutableList());

              ActionResult.Builder result = ActionResult.newBuilder();
              result.setExitCode(0);

              UploadManifest manifest =
                  new UploadManifest(
                      digestUtil,
                      remotePathResolver,
                      result,
                      remoteOptions.incompatibleRemoteSymlinks,
                      remoteOptions.allowSymlinkUpload);
              manifest.addFiles(outputFiles);
              manifest.setStdoutStderr(action.spawnExecutionContext.getFileOutErr());
              manifest.addAction(action.actionKey, action.action, action.command);
              if (manifest.getStderrDigest() != null) {
                result.setStderrDigest(manifest.getStderrDigest());
              }
              if (manifest.getStdoutDigest() != null) {
                result.setStdoutDigest(manifest.getStdoutDigest());
              }

              return Completable.concatArray(
                  uploadOutputs(remoteCache, action, manifest),
                  uploadActionResult(remoteCache, action, result.build()));
            },
            RemoteCache::release)
        .subscribeOn(Schedulers.io())
        .subscribe(reportUploadErrorObserver);

    uploadStartedLatch.await();
  }

  private final CompletableObserver reportUploadErrorObserver = new CompletableObserver() {
    @Override
    public void onSubscribe(@NonNull Disposable d) {

    }

    @Override
    public void onComplete() {
    }

    @Override
    public void onError(@NonNull Throwable e) {
      reportUploadError(e);
    }
  };

  private void reportUploadError(Throwable error) {
    if (remoteCacheInterrupted) {
      // If we interrupt manually, ignore cancellation errors.
      if (error instanceof CancellationException) {
        return;
      }
    }

    String errorMessage;
    if (!verboseFailures) {
       if (error instanceof IOException) {
         errorMessage = grpcAwareErrorMessage((IOException) error);
       } else {
         errorMessage = error.getMessage();
       }
    } else {
      // On --verbose_failures print the whole stack trace
      errorMessage = Throwables.getStackTraceAsString(error);
    }
    if (isNullOrEmpty(errorMessage)) {
      errorMessage = error.getClass().getSimpleName();
    }
    errorMessage = "Writing to Remote Cache:\n" + errorMessage;
    reporter.handle(Event.warn(errorMessage));
  }

  /**
   * Upload inputs of a remote action to remote cache if they are not presented already.
   *
   * <p>Must be called before calling {@link #executeRemotely}.
   */
  public void uploadInputsIfNotPresent(RemoteAction action)
      throws IOException, InterruptedException {
    checkState(!shutdown.get(), "shutdown");
    checkState(mayBeExecutedRemotely(action.spawn), "spawn can't be executed remotely");

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
  public RemoteActionResult executeRemotely(
      RemoteAction action, boolean acceptCachedResult, OperationObserver observer)
      throws IOException, InterruptedException {
    checkState(!shutdown.get(), "shutdown");
    checkState(mayBeExecutedRemotely(action.spawn), "spawn can't be executed remotely");

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
                  action.remoteActionExecutionContext,
                  serverLogs.lastLogPath,
                  e.getValue().getDigest()));
        }
      }
    }

    return serverLogs;
  }

  @Subscribe
  public void buildInterrupted(BuildInterruptedEvent event) {
    remoteCacheInterrupted = true;
  }

  /**
   * Shuts the service down. Wait for active network I/O to finish but new requests are rejected.
   */
  public void shutdown() {
    if (!shutdown.compareAndSet(false, true)) {
      return;
    }

    if (remoteCache != null) {
      remoteCache.release();

      if (remoteCacheInterrupted) {
        Thread.currentThread().interrupt();
      }

      try {
        remoteCache.awaitTermination();
      } catch (InterruptedException e) {
        remoteCacheInterrupted = true;
        reporter.handle(Event.warn("remote cache interrupted"));
        remoteCache.shutdownNow();
      }

      checkState(remoteCache.isClosed(), "remote cache is not closed properly.");
    }

    if (remoteExecutor != null) {
      remoteExecutor.close();
    }
  }
}

// Copyright 2025 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.Futures.transformAsync;
import static com.google.common.util.concurrent.Futures.whenAllSucceed;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.util.Utils.waitForBulkTransfer;
import static com.google.devtools.build.lib.unsafe.StringUnsafe.getInternalStringBytes;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.util.stream.Collectors.joining;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.OutputDirectory;
import build.bazel.remote.execution.v2.OutputFile;
import build.bazel.remote.execution.v2.Platform;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.base.Throwables;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.repository.DigestWriter;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext.CachePolicy;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.rules.repository.RepoRecordedInput;
import com.google.devtools.build.lib.runtime.RemoteRepoContentsCache;
import com.google.devtools.build.lib.unsafe.StringUnsafe;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.SortedMap;
import java.util.UUID;
import javax.annotation.Nullable;

/**
 * A cache for the contents of external repositories that is backed by an ordinary remote cache.
 *
 * <p>Upon a cache hit, the metadata of the files comprising the repository is downloaded and
 * injected into a {@link RemoteExternalOverlayFileSystem}. Downloads of file contents only occur
 * when Bazel needs to read a file (e.g., a BUILD or .bzl file) or if a file is an input to an
 * action executed locally. This can save both time taken to execute repo rules and compute file
 * digests and disk space required to store the contents of external repositories.
 *
 * <p>Repositories are cached as AC entries for a synthetic command with a special hash as the salt.
 * The contents are represented as an output file for the marker file and an output directory for
 * the contents.
 *
 * <p>If a repo rule has no dynamic dependencies, this hash is just the predeclared inputs hash
 * {@link DigestWriter}. If it has dynamic dependencies, then the AC entry for the predeclared
 * inputs hash will be an intermediate entry that lists one or more sets of {@link
 * RepoRecordedInput}s that a previously cached repo consumed during the evaluation of its rule. The
 * cache requests the current values of these inputs and computes the next hash to look up by a
 * rolling construction that combines the previous hash with the string representations of the
 * {@link RepoRecordedInput.WithValue}. This process is repeated until a final entry with the repo
 * contents is found or no matching entry exists.
 *
 * <p>By representing repos with dynamic dependencies as linked trees of AC entries, lookups are
 * efficient (they don't scale with the number of cached repos per predeclared inputs hash) and
 * regular LRU eviction policies remain effective for the most part. If a repo rule often requests
 * different inputs even with the same predeclared inputs hash and previously requested inputs and
 * values, it could result in large action results that grow over time. This is considered an
 * acceptable trade-off for simplicity for now and could be mitigated in the future by an explicit
 * GC mechanism such as "least recently added" eviction when the size of action result exceeds a
 * certain threshold.
 */
public final class RemoteRepoContentsCacheImpl implements RemoteRepoContentsCache {
  private static final UUID GUID = UUID.fromString("f4a165a9-5557-45a7-bf25-230b6d42393a");
  private static final String MARKER_FILE_PATH = ".recorded_inputs";
  private static final String REPO_DIRECTORY_PATH = "repo_contents";
  private static final Splitter SPLIT_ON_SPACE = Splitter.on(' ');

  private static final Command COMMAND =
      Command.newBuilder()
          // A unique but nonsensical command that is valid on all platforms. It is never executed,
          // but should pass all checks that an RE backend may apply to commands.
          .addArguments(GUID.toString())
          .addOutputPaths(MARKER_FILE_PATH)
          .addOutputPaths(REPO_DIRECTORY_PATH)
          .addOutputFiles(MARKER_FILE_PATH)
          .addOutputDirectories(REPO_DIRECTORY_PATH)
          .setPlatform(Platform.getDefaultInstance())
          .build();
  private static final ByteString COMMAND_BYTES = COMMAND.toByteString();
  private static final Directory INPUT_ROOT = Directory.getDefaultInstance();

  private final BlazeDirectories directories;
  private final CombinedCache cache;
  private final String buildRequestId;
  private final String commandId;
  private final boolean acceptCached;
  private final boolean uploadLocalResults;
  private final boolean verboseFailures;
  private final DigestUtil digestUtil;
  private final Action baseAction;
  private final Digest commandDigest;

  public RemoteRepoContentsCacheImpl(
      BlazeDirectories directories,
      CombinedCache cache,
      String buildRequestId,
      String commandId,
      boolean acceptCached,
      boolean uploadLocalResults,
      boolean verboseFailures) {
    this.directories = directories;
    this.cache = cache;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.acceptCached = acceptCached;
    this.uploadLocalResults = uploadLocalResults;
    this.verboseFailures = verboseFailures;
    this.digestUtil = cache.digestUtil;
    this.baseAction =
        Action.newBuilder()
            .setCommandDigest(digestUtil.compute(COMMAND))
            .setInputRootDigest(digestUtil.compute(INPUT_ROOT))
            .setPlatform(Platform.getDefaultInstance())
            .build();
    this.commandDigest = digestUtil.compute(COMMAND);
  }

  @Override
  public void addToCache(
      RepositoryName repoName,
      Path fetchedRepoDir,
      Path fetchedRepoMarkerFile,
      String predeclaredInputHash,
      ExtendedEventHandler reporter)
      throws InterruptedException {
    if (!(fetchedRepoDir.getFileSystem() instanceof RemoteExternalOverlayFileSystem)) {
      return;
    }
    var context = buildContext(repoName, CacheOp.UPLOAD);
    if (!context.getWriteCachePolicy().allowRemoteCache()) {
      return;
    }
    List<RepoRecordedInput.WithValue> recordedInputValues;
    try {
      var maybeRecordedInputValues =
          DigestWriter.readMarkerFile(
              FileSystemUtils.readContent(fetchedRepoMarkerFile, ISO_8859_1), predeclaredInputHash);
      if (maybeRecordedInputValues.isEmpty()) {
        return;
      }
      recordedInputValues = maybeRecordedInputValues.get();
    } catch (IOException e) {
      reporter.handle(
          Event.warn(
              "Failed to read marker file for repo %s, skipping: %s"
                  .formatted(repoName, maybeGetStackTrace(e))));
      return;
    }
    try {
      // TODO: Consider uploading asynchronously.
      var finalHash =
          uploadIntermediateActionResults(context, predeclaredInputHash, recordedInputValues);
      var action = buildAction(finalHash);
      var actionKey = new ActionKey(digestUtil.compute(action));
      var remotePathResolver = new RepoRemotePathResolver(fetchedRepoMarkerFile, fetchedRepoDir);
      var unused =
          UploadManifest.create(
                  cache.getRemoteCacheCapabilities(),
                  digestUtil,
                  remotePathResolver,
                  actionKey,
                  action,
                  COMMAND,
                  ImmutableList.of(fetchedRepoMarkerFile, fetchedRepoDir),
                  /* outErr= */ null,
                  /* exitCode= */ 0,
                  /* startTime= */ Instant.now(),
                  /* wallTimeInMs= */ 0,
                  /* preserveExecutableBit= */ true)
              .upload(context, cache, reporter);
    } catch (ExecException | IOException e) {
      reporter.handle(
          Event.warn(
              "Failed to upload repo contents to remote cache for repo %s: %s"
                  .formatted(repoName, maybeGetStackTrace(e))));
    }
  }

  @Override
  public boolean lookupCache(
      RepositoryName repoName,
      Path repoDir,
      String predeclaredInputHash,
      SkyFunction.Environment env)
      throws IOException, InterruptedException {
    try {
      return doLookupCache(repoName, repoDir, predeclaredInputHash, env);
    } catch (IOException e) {
      throw new IOException(
          "Failed to look up repo %s in the remote repo contents cache: %s"
              .formatted(repoName, maybeGetStackTrace(e)),
          e);
    }
  }

  private boolean doLookupCache(
      RepositoryName repoName,
      Path repoDir,
      String predeclaredInputHash,
      SkyFunction.Environment env)
      throws IOException, InterruptedException {
    if (!(repoDir.getFileSystem() instanceof RemoteExternalOverlayFileSystem remoteFs)) {
      return false;
    }

    var context = buildContext(repoName, CacheOp.DOWNLOAD);
    if (!context.getReadCachePolicy().allowRemoteCache()) {
      return false;
    }
    var finalEntry = fetchFinalCacheEntry(env, context, predeclaredInputHash);
    if (env.valuesMissing() || finalEntry == null) {
      return false;
    }

    ListenableFuture<byte[]> markerFileContentFuture;
    var markerFile = finalEntry.markerFile();
    // Inlining is an optional feature, so we have to be prepared to download the marker file.
    if (markerFile.getContents().isEmpty()) {
      markerFileContentFuture =
          cache.downloadBlob(
              context, MARKER_FILE_PATH, /* execPath= */ null, markerFile.getDigest());
    } else {
      markerFileContentFuture = immediateFuture(markerFile.getContents().toByteArray());
    }
    var repoDirectory = finalEntry.repoDirectory();
    var repoDirectoryContentFuture =
        transformAsync(
            cache.downloadBlob(
                context, REPO_DIRECTORY_PATH, /* execPath= */ null, repoDirectory.getTreeDigest()),
            (treeBytes) -> immediateFuture(Tree.parseFrom(treeBytes)),
            directExecutor());
    waitForBulkTransfer(ImmutableList.of(markerFileContentFuture, repoDirectoryContentFuture));

    String markerFileContent = new String(markerFileContentFuture.resultNow(), ISO_8859_1);
    var maybeRecordedInputs = DigestWriter.readMarkerFile(markerFileContent, predeclaredInputHash);
    if (maybeRecordedInputs.isEmpty()) {
      return false;
    }
    var outdatedReason =
        RepoRecordedInput.isAnyValueOutdated(env, directories, maybeRecordedInputs.get());
    if (env.valuesMissing() || outdatedReason.isPresent()) {
      env.getListener()
          .handle(
              Event.warn(
                  "Unexpectedly outdated cached repo %s: %s"
                      .formatted(repoName, outdatedReason.orElse("unknown reason"))));
      return false;
    }

    return remoteFs.injectRemoteRepo(
        repoName, repoDirectoryContentFuture.resultNow(), markerFileContent);
  }

  private enum CacheOp {
    DOWNLOAD,
    UPLOAD,
  }

  private RemoteActionExecutionContext buildContext(RepositoryName repoName, CacheOp cacheOp) {
    var metadata =
        TracingMetadataUtils.buildMetadata(
            buildRequestId, commandId, repoName.getName(), /* actionMetadata= */ null);
    // Don't upload local repo contents to the disk cache as the (local) `--repo_contents_cache` is
    // a better alternative for local caching. Do write through the disk cache for downloads from
    // the remote cache to speed up future usage.
    return RemoteActionExecutionContext.create(metadata)
        .withReadCachePolicy(acceptCached ? CachePolicy.ANY_CACHE : CachePolicy.NO_CACHE)
        .withWriteCachePolicy(
            switch (cacheOp) {
              case DOWNLOAD -> CachePolicy.ANY_CACHE;
              case UPLOAD ->
                  uploadLocalResults ? CachePolicy.REMOTE_CACHE_ONLY : CachePolicy.NO_CACHE;
            });
  }

  private Action buildAction(String inputHash) {
    // We choose to embed the hash into the salt simply because that results in a constant Command
    // message.
    return baseAction.toBuilder()
        .setSalt(ByteString.copyFrom(StringUnsafe.getByteArray(inputHash)))
        .build();
  }

  /**
   * Uploads the intermediate action results representing the inputs recorded at runtime and returns
   * the input hash to use for the final action result.
   */
  private String uploadIntermediateActionResults(
      RemoteActionExecutionContext context,
      String predeclaredInputHash,
      List<RepoRecordedInput.WithValue> recordedInputValues)
      throws IOException, InterruptedException {
    // The command is shared by all action results and small enough that FindMissingBlobs is not
    // worthwhile.
    waitForBulkTransfer(ImmutableSet.of(cache.uploadBlob(context, commandDigest, COMMAND_BYTES)));

    String rollingHash = predeclaredInputHash;
    var batches = RepoRecordedInput.WithValue.splitIntoBatches(recordedInputValues);
    var futures = new ArrayList<ListenableFuture<Void>>(batches.size());
    for (var batch : batches) {
      futures.add(
          addToActionResult(
              context,
              buildAction(rollingHash),
              Collections2.transform(batch, RepoRecordedInput.WithValue::input)));
      for (var recordedInputValue : batch) {
        rollingHash = rollForwardHash(rollingHash, recordedInputValue);
      }
    }
    waitForBulkTransfer(futures);
    return rollingHash;
  }

  /**
   * Adds the given set of recorded inputs as one of the alternative paths to the action result for
   * the given action, if not already present.
   *
   * <p>Most repo rule evaluations with a fixed previous batch of hashes (in particular, the same
   * predeclared inputs hash) will request a fixed set of inputs in the next batch. Thus, most
   * intermediate action results will only contain a single set of recorded inputs.
   */
  private ListenableFuture<Void> addToActionResult(
      RemoteActionExecutionContext context, Action action, Collection<RepoRecordedInput> inputs) {
    var actionKey = digestUtil.computeActionKey(action);
    var currentInputsFuture =
        Futures.transformAsync(
            cache.downloadActionResultAsync(
                context, actionKey, /* inlineOutErr= */ true, ImmutableSet.of()),
            (currentResult) -> {
              if (currentResult == null
                  || currentResult.actionResult().getStdoutDigest().getSizeBytes() == 0) {
                return immediateFuture("");
              }
              return fetchStdout(context, currentResult.actionResult());
            },
            directExecutor());
    return Futures.transformAsync(
        currentInputsFuture,
        currentInputsString -> {
          // RepoRecordedInput.toString() is guaranteed to return a string that doesn't contain
          // spaces or newlines. We can thus safely use spaces to separate inputs within a batch
          // and newlines to separate different batches.
          var newInputString =
              inputs.stream().map(RepoRecordedInput::toString).collect(joining(" "));
          if (currentInputsString.lines().anyMatch(newInputString::equals)) {
            // The current batch of inputs is already present, no need to update the action result.
            return immediateFuture(null);
          }
          // Add the new input to the top so that the most recently added inputs stay at the top.
          // This could be used to implement a simple "least recently added" eviction strategy in
          // the future in case the size of action results becomes a concern.
          //
          // Note that this update is inherently racy: multiple clients may add inputs concurrently,
          // resulting in some added inputs being lost since the REAPI does not provide a way to
          // update action results atomically. However, since different batches of inputs are
          // already rare and them being added concurrently even more so, the temporary loss of a
          // cache entry is an acceptable trade-off for simplicity.
          var newInputsString = newInputString + '\n' + currentInputsString;
          var stdoutBytes = getInternalStringBytes(newInputsString);
          var stdoutDigest = digestUtil.compute(stdoutBytes);
          var actionResult =
              ActionResult.newBuilder().setExitCode(0).setStdoutDigest(stdoutDigest).build();
          return transformAsync(
              whenAllSucceed(
                      cache.uploadBlob(context, actionKey.digest(), action.toByteString()),
                      cache.uploadBlob(context, stdoutDigest, ByteString.copyFrom(stdoutBytes)))
                  .run(() -> {}, directExecutor()),
              unused -> cache.uploadActionResult(context, actionKey, actionResult),
              directExecutor());
        },
        directExecutor());
  }

  /** Represents a single AC entry in the internal format used by the remote repo contents cache. */
  private sealed interface CacheEntry {
    /**
     * A final cache entry containing the contents of a repository.
     *
     * <p>Represented as an ActionResult with one output directory and one output file.
     *
     * @param repoDirectory the contents of the repository directory
     * @param markerFile the contents of the repository's marker file
     */
    record Final(OutputDirectory repoDirectory, OutputFile markerFile) implements CacheEntry {}

    /**
     * An intermediate cache entry that points to the keys of any number of further AC entries,
     * which can themselves be intermediate or final entries. The remote repo contents cache will
     * try them in order.
     *
     * @param nextInputHashes the keys under which the next AC entries should be looked up
     */
    record Intermediate(ImmutableList<String> nextInputHashes) implements CacheEntry {}

    /**
     * The cache entry didn't match any of the formats expected by this version of the remote repo
     * contents cache for the given human-readable reason.
     */
    record Invalid(String reason) implements CacheEntry {}
  }

  /**
   * Fetches a final cache entry for the given predeclared input hash by recursively following
   * intermediate entries if needed or returns null if no final entry could be found. The return
   * value must be ignored if {@link SkyFunction.Environment#valuesMissing()} is true.
   */
  @Nullable
  private CacheEntry.Final fetchFinalCacheEntry(
      SkyFunction.Environment env,
      RemoteActionExecutionContext context,
      String predeclaredInputHash)
      throws IOException, InterruptedException {
    var currentHashes = ImmutableList.of(predeclaredInputHash);
    while (!currentHashes.isEmpty()) {
      var nextHashes = ImmutableList.<String>builder();
      for (var hash : currentHashes) {
        switch (fetchCacheEntry(env, context, hash)) {
          case CacheEntry.Final finalEntry -> {
            return finalEntry;
          }
          case CacheEntry.Intermediate(ImmutableList<String> nextInputHashes) ->
              nextHashes.addAll(nextInputHashes);
          case CacheEntry.Invalid(String reason) -> env.getListener().handle(Event.warn(reason));
          case null -> {
            // Keep checking hashes to batch missing values in fewer restarts.
            Preconditions.checkState(env.valuesMissing());
          }
        }
      }
      if (env.valuesMissing()) {
        return null;
      }
      currentHashes = nextHashes.build();
    }
    return null;
  }

  // Returns null if and only if values are missing.
  @Nullable
  private CacheEntry fetchCacheEntry(
      SkyFunction.Environment env, RemoteActionExecutionContext context, String inputHash)
      throws IOException, InterruptedException {
    var actionKey = new ActionKey(digestUtil.compute(buildAction(inputHash)));
    // The marker file is read right after and thus requested to be inlined. If the action result
    // is an intermediate node, the full result will be contained in the stdout, which should thus
    // also be inlined.
    var cachedActionResult =
        cache.downloadActionResult(
            context, actionKey, /* inlineOutErr= */ true, ImmutableSet.of(MARKER_FILE_PATH));
    if (cachedActionResult == null) {
      return new CacheEntry.Intermediate(ImmutableList.of());
    }
    var actionResult = cachedActionResult.actionResult();

    if (actionResult.getExitCode() != 0) {
      return new CacheEntry.Invalid(
          "Unexpected exit code in action result for remotely cached repo %s:\n%s"
              .formatted(context.getRequestMetadata().getActionId(), actionResult));
    }
    if (actionResult.getOutputFilesCount() == 1
        && actionResult.getOutputDirectoriesCount() == 1
        && actionResult.getOutputSymlinksCount() == 0) {
      return new CacheEntry.Final(
          actionResult.getOutputDirectories(0), actionResult.getOutputFiles(0));
    }
    if (!(actionResult.getOutputFilesCount() == 0
        && actionResult.getOutputDirectoriesCount() == 0
        && actionResult.getOutputSymlinksCount() == 0
        && actionResult.getStdoutDigest().getSizeBytes() > 0)) {
      return new CacheEntry.Invalid(
          "Unexpected intermediate action result for remotely cached repo %s:\n%s"
              .formatted(context.getRequestMetadata().getActionId(), actionResult));
    }
    var stdoutFuture = fetchStdout(context, actionResult);
    waitForBulkTransfer(ImmutableList.of(stdoutFuture));

    // The action result's stdout contains multiple lines, each representing a batch of
    // RepoRecordedInputs separated by spaces. A given batch is valid only if all inputs in the
    // batch are, but separate batches are tried independently.
    var nextInputBatches =
        stdoutFuture
            .resultNow()
            .lines()
            .map(
                line ->
                    SPLIT_ON_SPACE
                        .splitToStream(line)
                        .map(RepoRecordedInput::parse)
                        .collect(toImmutableList()))
            .collect(toImmutableList());
    var uniqueNextInputs =
        nextInputBatches.stream().flatMap(List::stream).distinct().collect(toImmutableSet());
    RepoRecordedInput.prefetch(env, directories, uniqueNextInputs);
    if (env.valuesMissing()) {
      return null;
    }
    var nextHashes = ImmutableList.<String>builder();
    nextBatch:
    for (var batch : nextInputBatches) {
      var rollingHash = inputHash;
      for (var input : batch) {
        var value = input.getValue(env, directories);
        // Values have been prefetched above.
        Preconditions.checkState(!env.valuesMissing());
        if (!(value instanceof RepoRecordedInput.MaybeValue.Valid(String valueString))) {
          continue nextBatch;
        }
        rollingHash =
            rollForwardHash(rollingHash, new RepoRecordedInput.WithValue(input, valueString));
      }
      nextHashes.add(rollingHash);
    }
    return new CacheEntry.Intermediate(nextHashes.build());
  }

  private String rollForwardHash(String hash, RepoRecordedInput.WithValue inputWithValue) {
    return new Fingerprint()
        .addString(hash)
        .addString(inputWithValue.toString())
        .hexDigestAndReset();
  }

  private ListenableFuture<String> fetchStdout(
      RemoteActionExecutionContext context, ActionResult actionResult) {
    if (!actionResult.getStdoutRaw().isEmpty()) {
      return immediateFuture(
          StringUnsafe.newInstance(actionResult.getStdoutRaw().toByteArray(), StringUnsafe.LATIN1));
    }
    return Futures.transform(
        cache.downloadBlob(context, actionResult.getStdoutDigest()),
        stdout -> StringUnsafe.newInstance(stdout, StringUnsafe.LATIN1),
        directExecutor());
  }

  private String maybeGetStackTrace(Exception e) {
    return verboseFailures ? Throwables.getStackTraceAsString(e) : e.getMessage();
  }

  private record RepoRemotePathResolver(Path fetchedRepoMarkerFile, Path fetchedRepoDir)
      implements RemotePathResolver {

    @Override
    public String localPathToOutputPath(Path path) {
      // Map repo marker file and contents to fixed locations under the fake remote exec root.
      if (path.equals(fetchedRepoMarkerFile)) {
        return MARKER_FILE_PATH;
      }
      if (path.equals(fetchedRepoDir)) {
        return REPO_DIRECTORY_PATH;
      }
      return REPO_DIRECTORY_PATH + "/" + path.relativeTo(fetchedRepoDir).getPathString();
    }

    @Override
    public String localPathToOutputPath(PathFragment execPath) {
      throw new UnsupportedOperationException("Not used");
    }

    @Override
    public PathFragment getWorkingDirectory() {
      throw new UnsupportedOperationException("Not used");
    }

    @Override
    public Path outputPathToLocalPath(String outputPath) {
      throw new UnsupportedOperationException("Not used");
    }

    @Override
    public PathFragment localPathToExecPath(PathFragment localPath) {
      throw new UnsupportedOperationException("Not used");
    }

    @Override
    public SortedMap<PathFragment, ActionInput> getInputMapping(
        SpawnRunner.SpawnExecutionContext context, boolean willAccessRepeatedly) {
      throw new UnsupportedOperationException("Not used");
    }
  }
}

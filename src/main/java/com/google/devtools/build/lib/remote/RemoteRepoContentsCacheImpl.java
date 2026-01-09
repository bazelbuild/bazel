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
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.util.Utils.waitForBulkTransfer;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.base.Splitter;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ExecException;
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
import com.google.devtools.build.lib.runtime.RemoteRepoContentsCache;
import com.google.devtools.build.lib.unsafe.StringUnsafe;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.SortedMap;
import java.util.UUID;
import java.util.concurrent.ExecutionException;

/**
 * A cache for the contents of external repositories that is backed by an ordinary remote cache.
 *
 * <p>Upon a cache hit, the metadata of the files comprising the repository is downloaded and
 * injected into a {@link RemoteExternalOverlayFileSystem}. Downloads of file contents only occur
 * when Bazel needs to read a file (e.g., a BUILD or .bzl file) or if a file is an input to an
 * action executed locally. This can save both time taken to execute repo rules and compute file
 * digests and disk space required to store the contents of external repositories.
 *
 * <p>Repositories are cached as AC entries for a synthetic command with the predeclared input hash
 * as the salt. The contents are represented as an output file for the marker file and an output
 * directory for the contents.
 *
 * <p>At this point the cache only supports repository rules with no dependencies expressed at
 * runtime. Verifying whether such dependencies are up to date can't be done via a single hash as
 * the set of dependencies is not known ahead of time. Support for such rules would require a
 * two-stage cache lookup in which the first lookup may produce multiple marker files.
 */
public final class RemoteRepoContentsCacheImpl implements RemoteRepoContentsCache {
  private static final UUID GUID = UUID.fromString("f4a165a9-5557-45a7-bf25-230b6d42393a");
  private static final String MARKER_FILE_PATH = ".recorded_inputs";
  private static final String REPO_DIRECTORY_PATH = "repo_contents";

  private static final Command COMMAND =
      Command.newBuilder()
          // A unique but nonsensical command that is valid on all platforms. It is never executed,
          // but should pass all checks that an RE backend may apply to commands.
          .addArguments(GUID.toString())
          .addOutputPaths(MARKER_FILE_PATH)
          .addOutputPaths(REPO_DIRECTORY_PATH)
          .addOutputFiles(MARKER_FILE_PATH)
          .addOutputDirectories(REPO_DIRECTORY_PATH)
          .build();
  private static final Directory INPUT_ROOT = Directory.getDefaultInstance();

  private final CombinedCache cache;
  private final String buildRequestId;
  private final String commandId;
  private final boolean acceptCached;
  private final boolean uploadLocalResults;
  private final boolean verboseFailures;
  private final DigestUtil digestUtil;
  private final Action baseAction;

  public RemoteRepoContentsCacheImpl(
      CombinedCache cache,
      String buildRequestId,
      String commandId,
      boolean acceptCached,
      boolean uploadLocalResults,
      boolean verboseFailures) {
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.cache = cache;
    this.acceptCached = acceptCached;
    this.uploadLocalResults = uploadLocalResults;
    this.verboseFailures = verboseFailures;
    this.digestUtil = cache.digestUtil;
    this.baseAction =
        Action.newBuilder()
            .setCommandDigest(digestUtil.compute(COMMAND))
            .setInputRootDigest(digestUtil.compute(INPUT_ROOT))
            .build();
  }

  @Override
  public void addToCache(
      RepositoryName repoName,
      Path fetchedRepoDir,
      Path fetchedRepoMarkerFile,
      String predeclaredInputHash,
      ExtendedEventHandler reporter)
      throws InterruptedException {
    var context = buildContext(repoName);
    if (!context.getWriteCachePolicy().allowRemoteCache()) {
      return;
    }
    try {
      if (FileSystemUtils.readLinesAsLatin1(fetchedRepoMarkerFile).stream()
              .filter(line -> !line.isEmpty())
              .count()
          != 1) {
        // This cache currently only supports marker files that contain nothing but the predeclared
        // inputs hash. Repo rules with dependencies expressed only at runtime would require a
        // two-stage cache lookup. Among the rules that are supported are http_archive and
        // git_repository without patches.
        return;
      }
    } catch (IOException e) {
      reporter.handle(
          Event.warn(
              "Failed to read marker file repo %s, skipping: %s"
                  .formatted(repoName, maybeGetStackTrace(e))));
    }
    var action = buildAction(predeclaredInputHash);
    var actionKey = new ActionKey(digestUtil.compute(action));
    var remotePathResolver = new RepoRemotePathResolver(fetchedRepoMarkerFile, fetchedRepoDir);
    try {
      // TODO: Consider uploading asynchronously.
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
      ExtendedEventHandler reporter)
      throws IOException, InterruptedException {
    try {
      return doLookupCache(repoName, repoDir, predeclaredInputHash, reporter);
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
      ExtendedEventHandler reporter)
      throws IOException, InterruptedException {
    if (!(repoDir.getFileSystem() instanceof RemoteExternalOverlayFileSystem remoteFs)) {
      return false;
    }

    var context = buildContext(repoName);
    if (!context.getReadCachePolicy().allowRemoteCache()) {
      return false;
    }
    var actionKey = new ActionKey(digestUtil.compute(buildAction(predeclaredInputHash)));
    // The marker file is read right after and thus requested to be inlined.
    var cachedActionResult =
        cache.downloadActionResult(
            context, actionKey, /* inlineOutErr= */ false, ImmutableSet.of(MARKER_FILE_PATH));
    if (cachedActionResult == null) {
      return false;
    }

    var actionResult = cachedActionResult.actionResult();
    if (actionResult.getExitCode() != 0
        || actionResult.getOutputFilesCount() != 1
        || actionResult.getOutputDirectoriesCount() != 1) {
      reporter.handle(
          Event.warn(
              String.format(
                  "Unexpected action result for cached repo %s: exit code %d, %d output files, %d"
                      + " output directories",
                  repoName,
                  actionResult.getExitCode(),
                  actionResult.getOutputFilesCount(),
                  actionResult.getOutputDirectoriesCount())));
      return false;
    }

    ListenableFuture<byte[]> markerFileContentFuture;
    var markerFile = actionResult.getOutputFiles(0);
    // Inlining is an optional feature, so we have to be prepared to download the marker file.
    if (markerFile.getContents().isEmpty()) {
      markerFileContentFuture =
          cache.downloadBlob(
              context, MARKER_FILE_PATH, /* execPath= */ null, markerFile.getDigest());
    } else {
      markerFileContentFuture = immediateFuture(markerFile.getContents().toByteArray());
    }
    var repoDirectory = actionResult.getOutputDirectories(0);
    var repoDirectoryContentFuture =
        Futures.transformAsync(
            cache.downloadBlob(
                context, REPO_DIRECTORY_PATH, /* execPath= */ null, repoDirectory.getTreeDigest()),
            (treeBytes) -> immediateFuture(Tree.parseFrom(treeBytes)),
            directExecutor());
    waitForBulkTransfer(ImmutableList.of(markerFileContentFuture, repoDirectoryContentFuture));
    String markerFileContent;
    Tree repoDirectoryContent;
    try {
      markerFileContent = new String(markerFileContentFuture.get(), StandardCharsets.ISO_8859_1);
      repoDirectoryContent = repoDirectoryContentFuture.get();
    } catch (ExecutionException e) {
      throw new IllegalStateException(
          "waitForBulkTransfer should have thrown: " + maybeGetStackTrace(e));
    }
    var markerFileLines =
        Splitter.on('\n')
            .splitToStream(markerFileContent)
            .filter(line -> !line.isEmpty())
            .collect(toImmutableList());
    if (markerFileLines.size() > 1) {
      reporter.handle(
          Event.warn(
              "Marker file for repo %s has extra lines, skipping:\n%s"
                  .formatted(
                      repoName,
                      String.join("\n", markerFileLines.subList(1, markerFileLines.size())))));
      return false;
    }
    if (!markerFileLines.getFirst().equals(predeclaredInputHash)) {
      reporter.handle(
          Event.warn(
              "Predeclared input hash mismatch for repo %s: expected %s, got %s"
                  .formatted(repoName, predeclaredInputHash, markerFileLines.getFirst())));
      return false;
    }

    return remoteFs.injectRemoteRepo(repoName, repoDirectoryContent, markerFileContent);
  }

  private RemoteActionExecutionContext buildContext(RepositoryName repoName) {
    var metadata =
        TracingMetadataUtils.buildMetadata(
            buildRequestId, commandId, repoName.getName(), /* actionMetadata= */ null);
    // Don't use the disk cache as `--repo_contents_cache` is a strictly better alternative for
    // local caching.
    return RemoteActionExecutionContext.create(metadata)
        .withReadCachePolicy(acceptCached ? CachePolicy.REMOTE_CACHE_ONLY : CachePolicy.NO_CACHE)
        .withWriteCachePolicy(
            uploadLocalResults ? CachePolicy.REMOTE_CACHE_ONLY : CachePolicy.NO_CACHE);
  }

  private Action buildAction(String predeclaredInputHash) {
    // The predeclared input hash uniquely identifies the repo rule and all its attributes, but not
    // dependencies established at runtime. We choose to embed it into the salt simply because that
    // results in a constant Command message.
    return baseAction.toBuilder()
        .setSalt(ByteString.copyFrom(StringUnsafe.getByteArray(predeclaredInputHash)))
        .build();
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

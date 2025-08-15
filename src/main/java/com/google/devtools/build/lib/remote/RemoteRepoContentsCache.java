package com.google.devtools.build.lib.remote;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.util.Utils.waitForBulkTransfer;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.runtime.RepoContentsCache;
import com.google.devtools.build.lib.unsafe.StringUnsafe;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.SortedMap;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.regex.Pattern;

public final class RemoteRepoContentsCache implements RepoContentsCache {
  private static final UUID GUID = UUID.fromString("f4a165a9-5557-45a7-bf25-230b6d42393a");
  private static final String MARKER_FILE_PATH = ".recorded_inputs";
  private static final String REPO_DIRECTORY_PATH = "repo_contents";

  private static final Command COMMAND =
      Command.newBuilder()
          .addArguments(GUID.toString())
          .addOutputPaths(MARKER_FILE_PATH)
          .addOutputPaths(REPO_DIRECTORY_PATH)
          .addOutputFiles(MARKER_FILE_PATH)
          .addOutputDirectories(REPO_DIRECTORY_PATH)
          .build();
  private static final Directory INPUT_ROOT = Directory.getDefaultInstance();

  private final String buildRequestId;
  private final String commandId;
  private final CombinedCache cache;
  private final DigestUtil digestUtil;
  private final Action baseAction;

  public RemoteRepoContentsCache(CombinedCache cache, String buildRequestId, String commandId) {
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.cache = cache;
    this.digestUtil = cache.digestUtil;
    this.baseAction =
        Action.newBuilder()
            .setCommandDigest(digestUtil.compute(COMMAND))
            .setInputRootDigest(digestUtil.compute(INPUT_ROOT))
            .build();
  }

  @Override
  public void addToCache(
      RepositoryName repoName, Path fetchedRepoDir,
      Path fetchedRepoMarkerFile,
      String predeclaredInputHash,
      ExtendedEventHandler reporter)
      throws InterruptedException {
    var context = buildContext(repoName);
    var action = buildAction(predeclaredInputHash);
    var actionKey = new ActionKey(digestUtil.compute(action));
    var remotePathResolver =
        new RemotePathResolver() {
          @Override
          public String localPathToOutputPath(Path path) {
            if (path.equals(fetchedRepoMarkerFile)) {
              return MARKER_FILE_PATH;
            }
            if (path.equals(fetchedRepoDir)) {
              return REPO_DIRECTORY_PATH;
            }
            return REPO_DIRECTORY_PATH + "/" + path.relativeTo(fetchedRepoDir).getPathString();
          }

          @Override
          public String getWorkingDirectory() {
            throw new UnsupportedOperationException("Not used");
          }

          @Override
          public String localPathToOutputPath(PathFragment execPath) {
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

          @Override
          public void walkInputs(
              Spawn spawn,
              SpawnRunner.SpawnExecutionContext context,
              SpawnInputExpander.InputVisitor visitor) {
            throw new UnsupportedOperationException("Not used");
          }
        };
    try {
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
              /* wallTimeMs= */ 0)
          .upload(context, cache, reporter);
    } catch (ExecException | IOException e) {
      reporter.handle(
          Event.warn(
              "Failed to upload repo contents to remote cache for repo %s: %s"
                  .formatted(context.getRequestMetadata().getActionId(), e.getMessage())));
    }
  }

  @Override
  public boolean lookupCache(
      RepositoryName repoName,
      Path repoDir,
      String predeclaredInputHash,
      ExtendedEventHandler reporter)
      throws IOException, InterruptedException {
    if (!(repoDir.getFileSystem() instanceof RemoteOverlayFileSystem remoteFs)) {
      return false;
    }

    var context = buildContext(repoName);
    var actionKey = new ActionKey(digestUtil.compute(buildAction(predeclaredInputHash)));
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
      throw new IOException(
          String.format(
              "Unexpected action result for cached repo %s: exit code %d, %d output files, %d output directories",
              context.getRequestMetadata().getActionId(),
              actionResult.getExitCode(),
              actionResult.getOutputFilesCount(),
              actionResult.getOutputDirectoriesCount()));
    }

    ListenableFuture<byte[]> markerFileContentFuture;
    var markerFile = actionResult.getOutputFiles(0);
    if (markerFile.getContents().isEmpty()) {
      markerFileContentFuture =
          cache.downloadBlob(
              context, MARKER_FILE_PATH, /* execPath= */ null, markerFile.getDigest());
    } else {
      markerFileContentFuture = Futures.immediateFuture(markerFile.getContents().toByteArray());
    }
    var repoDirectory = actionResult.getOutputDirectories(0);
    var repoDirectoryContentFuture =
        Futures.transformAsync(
            cache.downloadBlob(
                context, REPO_DIRECTORY_PATH, /* execPath= */ null, repoDirectory.getTreeDigest()),
            (treeBytes) -> immediateFuture(Tree.parseFrom(treeBytes)),
            directExecutor());
    waitForBulkTransfer(
        ImmutableList.of(markerFileContentFuture, repoDirectoryContentFuture),
        /* cancelRemainingOnInterrupt= */ true);
    String markerFileContent;
    Tree repoDirectoryContent;
    try {
      markerFileContent = new String(markerFileContentFuture.get(), StandardCharsets.ISO_8859_1);
      repoDirectoryContent = repoDirectoryContentFuture.get();
    } catch (ExecutionException e) {
      throw new IllegalStateException("waitForBulkTransfer should have thrown", e);
    }
    var markerFileLines =
        Pattern.compile("\n")
            .splitAsStream(markerFileContent)
            .filter(line -> !line.isEmpty())
            .collect(toImmutableList());
    if (markerFileLines.size() > 1) {
      reporter.handle(
          Event.info(
              "Marker file for repo %s has extra lines:\n%s"
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

    remoteFs.injectRemoteRepo(repoName, repoDirectoryContent);
    return true;
  }

  private RemoteActionExecutionContext buildContext(RepositoryName repoName) {
    var metadata =
        TracingMetadataUtils.buildMetadata(
            buildRequestId, commandId, repoName.getName(), /* actionExecutionMetadata= */ null);
    return RemoteActionExecutionContext.create(metadata)
        .withReadCachePolicy(RemoteActionExecutionContext.CachePolicy.REMOTE_CACHE_ONLY)
        .withWriteCachePolicy(RemoteActionExecutionContext.CachePolicy.REMOTE_CACHE_ONLY);
  }

  private Action buildAction(String predeclaredInputHash) {
    return baseAction.toBuilder()
        .setSalt(ByteString.copyFrom(StringUnsafe.getByteArray(predeclaredInputHash)))
        .build();
  }
}

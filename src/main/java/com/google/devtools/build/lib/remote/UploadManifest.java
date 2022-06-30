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

import static com.google.common.base.Throwables.throwIfInstanceOf;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.util.RxFutures.toCompletable;
import static com.google.devtools.build.lib.remote.util.RxFutures.toSingle;
import static com.google.devtools.build.lib.remote.util.RxUtils.mergeBulkTransfer;
import static com.google.devtools.build.lib.remote.util.RxUtils.toTransferResult;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionUploadFinishedEvent;
import com.google.devtools.build.lib.actions.ActionUploadStartedEvent;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.RxUtils;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.RemoteExecution;
import com.google.devtools.build.lib.server.FailureDetails.RemoteExecution.Code;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.protobuf.ByteString;
import com.google.protobuf.Timestamp;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.core.Single;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** UploadManifest adds output metadata to a {@link ActionResult}. */
public class UploadManifest {

  private final DigestUtil digestUtil;
  private final RemotePathResolver remotePathResolver;
  private final ActionResult.Builder result;
  private final boolean allowSymlinks;
  private final boolean uploadSymlinks;
  private final Map<Digest, Path> digestToFile = new HashMap<>();
  private final Map<Digest, ByteString> digestToBlobs = new HashMap<>();
  @Nullable private ActionKey actionKey;
  private Digest stderrDigest;
  private Digest stdoutDigest;

  public static UploadManifest create(
      RemoteOptions remoteOptions,
      DigestUtil digestUtil,
      RemotePathResolver remotePathResolver,
      ActionKey actionKey,
      Action action,
      Command command,
      Collection<Path> outputFiles,
      FileOutErr outErr,
      int exitCode,
      Optional<Instant> startTime,
      Optional<Duration> wallTime)
      throws ExecException, IOException {
    ActionResult.Builder result = ActionResult.newBuilder();
    result.setExitCode(exitCode);

    UploadManifest manifest =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            remoteOptions.incompatibleRemoteSymlinks,
            remoteOptions.allowSymlinkUpload);
    manifest.addFiles(outputFiles);
    manifest.setStdoutStderr(outErr);
    manifest.addAction(actionKey, action, command);
    if (manifest.getStderrDigest() != null) {
      result.setStderrDigest(manifest.getStderrDigest());
    }
    if (manifest.getStdoutDigest() != null) {
      result.setStdoutDigest(manifest.getStdoutDigest());
    }

    if (startTime.isPresent() && wallTime.isPresent()) {
      result
          .getExecutionMetadataBuilder()
          .setWorkerStartTimestamp(instantToTimestamp(startTime.get()))
          .setWorkerCompletedTimestamp(instantToTimestamp(startTime.get().plus(wallTime.get())));
    }

    return manifest;
  }

  private static Timestamp instantToTimestamp(Instant instant) {
    return Timestamp.newBuilder()
        .setSeconds(instant.getEpochSecond())
        .setNanos(instant.getNano())
        .build();
  }

  /**
   * Create an UploadManifest from an ActionResult builder and an exec root. The ActionResult
   * builder is populated through a call to {@link #addFile(Digest, Path)}.
   */
  @VisibleForTesting
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

  private void setStdoutStderr(FileOutErr outErr) throws IOException {
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
  @VisibleForTesting
  void addFiles(Collection<Path> files) throws ExecException, IOException {
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
        PathFragment target = PathFragment.create(file.readSymbolicLink());
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
  private void addAction(RemoteCacheClient.ActionKey actionKey, Action action, Command command) {
    Preconditions.checkState(this.actionKey == null, "Already added an action");
    this.actionKey = actionKey;
    digestToBlobs.put(actionKey.getDigest(), action.toByteString());
    digestToBlobs.put(action.getCommandDigest(), command.toByteString());
  }

  /** Map of digests to file paths to upload. */
  public Map<Digest, Path> getDigestToFile() {
    return digestToFile;
  }

  /**
   * Map of digests to chunkers to upload. When the file is a regular, non-directory file it is
   * transmitted through {@link #getDigestToFile()}. When it is a directory, it is transmitted as a
   * {@link Tree} protobuf message through {@link #getDigestToBlobs()}.
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

  private void addFileSymbolicLink(Path file, PathFragment target) {
    result
        .addOutputFileSymlinksBuilder()
        .setPath(remotePathResolver.localPathToOutputPath(file))
        .setTarget(target.toString());
  }

  private void addDirectorySymbolicLink(Path file, PathFragment target) {
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
        // The permission of output file is changed to 0555 after action execution
        .setIsExecutable(true);

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
        PathFragment target = PathFragment.create(child.readSymbolicLink());
        if (uploadSymlinks && !target.isAbsolute()) {
          // Whether it is dangling or not, we're passing it on.
          b.addSymlinksBuilder().setName(name).setTarget(target.toString());
          continue;
        }
        // Need to resolve the symbolic link now to know whether to upload a file or a directory.
        FileStatus statFollow = child.statIfFound(Symlinks.FOLLOW);
        if (statFollow == null) {
          throw new IOException(
              String.format("Action output %s is a dangling symbolic link to %s ", child, target));
        }
        if (statFollow.isFile() && !statFollow.isSpecialFile()) {
          Digest digest = digestUtil.compute(child);
          b.addFilesBuilder().setName(name).setDigest(digest).setIsExecutable(child.isExecutable());
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

    FailureDetail failureDetail =
        FailureDetail.newBuilder()
            .setMessage(message)
            .setRemoteExecution(RemoteExecution.newBuilder().setCode(Code.ILLEGAL_OUTPUT))
            .build();
    throw new UserExecException(failureDetail);
  }

  @VisibleForTesting
  ActionResult getActionResult() {
    return result.build();
  }

  /** Uploads outputs and action result (if exit code is 0) to remote cache. */
  public ActionResult upload(
      RemoteActionExecutionContext context, RemoteCache remoteCache, ExtendedEventHandler reporter)
      throws IOException, InterruptedException, ExecException {
    try {
      return uploadAsync(context, remoteCache, reporter).blockingGet();
    } catch (RuntimeException e) {
      Throwable cause = e.getCause();
      if (cause != null) {
        throwIfInstanceOf(cause, InterruptedException.class);
        throwIfInstanceOf(cause, IOException.class);
        throwIfInstanceOf(cause, ExecException.class);
      }
      throw e;
    }
  }

  private Completable upload(
      RemoteActionExecutionContext context, RemoteCache remoteCache, Digest digest) {
    Path file = digestToFile.get(digest);
    if (file != null) {
      return toCompletable(() -> remoteCache.uploadFile(context, digest, file), directExecutor());
    }

    ByteString blob = digestToBlobs.get(digest);
    if (blob == null) {
      String message = "FindMissingBlobs call returned an unknown digest: " + digest;
      return Completable.error(new IOException(message));
    }

    return toCompletable(() -> remoteCache.uploadBlob(context, digest, blob), directExecutor());
  }

  private static void reportUploadStarted(
      ExtendedEventHandler reporter,
      @Nullable ActionExecutionMetadata action,
      String prefix,
      Iterable<Digest> digests) {
    if (action != null) {
      for (Digest digest : digests) {
        reporter.post(ActionUploadStartedEvent.create(action, prefix + digest.getHash()));
      }
    }
  }

  private static void reportUploadFinished(
      ExtendedEventHandler reporter,
      @Nullable ActionExecutionMetadata action,
      String resourceIdPrefix,
      Iterable<Digest> digests) {
    if (action != null) {
      for (Digest digest : digests) {
        reporter.post(
            ActionUploadFinishedEvent.create(action, resourceIdPrefix + digest.getHash()));
      }
    }
  }

  /**
   * Returns a {@link Single} which upon subscription will upload outputs and action result (if exit
   * code is 0) to remote cache.
   */
  public Single<ActionResult> uploadAsync(
      RemoteActionExecutionContext context,
      RemoteCache remoteCache,
      ExtendedEventHandler reporter) {
    Collection<Digest> digests = new ArrayList<>();
    digests.addAll(digestToFile.keySet());
    digests.addAll(digestToBlobs.keySet());

    ActionExecutionMetadata action = context.getSpawnOwner();

    String outputPrefix = "cas/";
    Flowable<RxUtils.TransferResult> bulkTransfers =
        toSingle(() -> remoteCache.findMissingDigests(context, digests), directExecutor())
            .doOnSubscribe(d -> reportUploadStarted(reporter, action, outputPrefix, digests))
            .doOnError(error -> reportUploadFinished(reporter, action, outputPrefix, digests))
            .doOnDispose(() -> reportUploadFinished(reporter, action, outputPrefix, digests))
            .doOnSuccess(
                missingDigests -> {
                  List<Digest> existedDigests =
                      digests.stream()
                          .filter(digest -> !missingDigests.contains(digest))
                          .collect(Collectors.toList());
                  reportUploadFinished(reporter, action, outputPrefix, existedDigests);
                })
            .flatMapPublisher(Flowable::fromIterable)
            .flatMapSingle(
                digest ->
                    toTransferResult(upload(context, remoteCache, digest))
                        .doFinally(
                            () ->
                                reportUploadFinished(
                                    reporter, action, outputPrefix, ImmutableList.of(digest))));
    Completable uploadOutputs = mergeBulkTransfer(bulkTransfers);

    ActionResult actionResult = result.build();
    Completable uploadActionResult = Completable.complete();
    if (actionResult.getExitCode() == 0 && actionKey != null) {
      String actionResultPrefix = "ac/";
      uploadActionResult =
          toCompletable(
                  () -> remoteCache.uploadActionResult(context, actionKey, actionResult),
                  directExecutor())
              .doOnSubscribe(
                  d ->
                      reportUploadStarted(
                          reporter,
                          action,
                          actionResultPrefix,
                          ImmutableList.of(actionKey.getDigest())))
              .doFinally(
                  () ->
                      reportUploadFinished(
                          reporter,
                          action,
                          actionResultPrefix,
                          ImmutableList.of(actionKey.getDigest())));
    }

    return Completable.concatArray(uploadOutputs, uploadActionResult).toSingleDefault(actionResult);
  }
}

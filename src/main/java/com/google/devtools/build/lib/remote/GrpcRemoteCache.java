// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.bytestream.ByteStreamGrpc;
import com.google.bytestream.ByteStreamGrpc.ByteStreamBlockingStub;
import com.google.bytestream.ByteStreamProto.ReadRequest;
import com.google.bytestream.ByteStreamProto.ReadResponse;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.remote.Digests.ActionKey;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.remoteexecution.v1test.ActionCacheGrpc;
import com.google.devtools.remoteexecution.v1test.ActionCacheGrpc.ActionCacheBlockingStub;
import com.google.devtools.remoteexecution.v1test.ActionResult;
import com.google.devtools.remoteexecution.v1test.Command;
import com.google.devtools.remoteexecution.v1test.ContentAddressableStorageGrpc;
import com.google.devtools.remoteexecution.v1test.ContentAddressableStorageGrpc.ContentAddressableStorageBlockingStub;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.devtools.remoteexecution.v1test.Directory;
import com.google.devtools.remoteexecution.v1test.FindMissingBlobsRequest;
import com.google.devtools.remoteexecution.v1test.FindMissingBlobsResponse;
import com.google.devtools.remoteexecution.v1test.GetActionResultRequest;
import com.google.devtools.remoteexecution.v1test.OutputFile;
import com.google.devtools.remoteexecution.v1test.UpdateActionResultRequest;
import io.grpc.CallCredentials;
import io.grpc.Channel;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/** A RemoteActionCache implementation that uses gRPC calls to a remote cache server. */
@ThreadSafe
public class GrpcRemoteCache implements RemoteActionCache {
  private final RemoteOptions options;
  private final CallCredentials credentials;
  private final Channel channel;
  private final Retrier retrier;

  private final ByteStreamUploader uploader;

  private final ListeningScheduledExecutorService retryScheduler =
      MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));

  @VisibleForTesting
  public GrpcRemoteCache(Channel channel, CallCredentials credentials, RemoteOptions options,
      Retrier retrier) {
    this.options = options;
    this.credentials = credentials;
    this.channel = channel;
    this.retrier = retrier;

    uploader = new ByteStreamUploader(options.remoteInstanceName, channel, credentials,
        options.remoteTimeout, retrier, retryScheduler);
  }

  private ContentAddressableStorageBlockingStub casBlockingStub() {
    return ContentAddressableStorageGrpc.newBlockingStub(channel)
        .withInterceptors(TracingMetadataUtils.attachMetadataFromContextInterceptor())
        .withCallCredentials(credentials)
        .withDeadlineAfter(options.remoteTimeout, TimeUnit.SECONDS);
  }

  private ByteStreamBlockingStub bsBlockingStub() {
    return ByteStreamGrpc.newBlockingStub(channel)
        .withInterceptors(TracingMetadataUtils.attachMetadataFromContextInterceptor())
        .withCallCredentials(credentials)
        .withDeadlineAfter(options.remoteTimeout, TimeUnit.SECONDS);
  }

  private ActionCacheBlockingStub acBlockingStub() {
    return ActionCacheGrpc.newBlockingStub(channel)
        .withInterceptors(TracingMetadataUtils.attachMetadataFromContextInterceptor())
        .withCallCredentials(credentials)
        .withDeadlineAfter(options.remoteTimeout, TimeUnit.SECONDS);
  }

  @Override
  public void close() {
    retryScheduler.shutdownNow();
    uploader.shutdown();
  }

  public static boolean isRemoteCacheOptions(RemoteOptions options) {
    return options.remoteCache != null;
  }

  private ImmutableSet<Digest> getMissingDigests(Iterable<Digest> digests)
      throws IOException, InterruptedException {
    FindMissingBlobsRequest.Builder request =
        FindMissingBlobsRequest.newBuilder()
            .setInstanceName(options.remoteInstanceName)
            .addAllBlobDigests(digests);
    if (request.getBlobDigestsCount() == 0) {
      return ImmutableSet.of();
    }
    FindMissingBlobsResponse response =
        retrier.execute(() -> casBlockingStub().findMissingBlobs(request.build()));
    return ImmutableSet.copyOf(response.getMissingBlobDigestsList());
  }

  /**
   * Upload enough of the tree metadata and data into remote cache so that the entire tree can be
   * reassembled remotely using the root digest.
   */
  @Override
  public void ensureInputsPresent(
      TreeNodeRepository repository, Path execRoot, TreeNode root, Command command)
      throws IOException, InterruptedException {
    repository.computeMerkleDigests(root);
    Digest commandDigest = Digests.computeDigest(command);
    // TODO(olaola): avoid querying all the digests, only ask for novel subtrees.
    ImmutableSet<Digest> missingDigests =
        getMissingDigests(
            Iterables.concat(repository.getAllDigests(root), ImmutableList.of(commandDigest)));

    List<Chunker> toUpload = new ArrayList<>();
    // Only upload data that was missing from the cache.
    ArrayList<ActionInput> missingActionInputs = new ArrayList<>();
    ArrayList<Directory> missingTreeNodes = new ArrayList<>();
    HashSet<Digest> missingTreeDigests = new HashSet<>(missingDigests);
    missingTreeDigests.remove(commandDigest);
    repository.getDataFromDigests(missingTreeDigests, missingActionInputs, missingTreeNodes);

    if (missingDigests.contains(commandDigest)) {
      toUpload.add(new Chunker(command.toByteArray()));
    }
    if (!missingTreeNodes.isEmpty()) {
      for (Directory d : missingTreeNodes) {
        toUpload.add(new Chunker(d.toByteArray()));
      }
    }
    if (!missingActionInputs.isEmpty()) {
      MetadataProvider inputFileCache = repository.getInputFileCache();
      for (ActionInput actionInput : missingActionInputs) {
        toUpload.add(new Chunker(actionInput, inputFileCache, execRoot));
      }
    }
    uploader.uploadBlobs(toUpload);
  }

  /**
   * Download all results of a remotely executed action locally. TODO(olaola): will need to amend to
   * include the {@link com.google.devtools.build.lib.remote.TreeNodeRepository} for updating.
   */
  @Override
  public void download(ActionResult result, Path execRoot, FileOutErr outErr)
      throws ExecException, IOException, InterruptedException {
    try {
      for (OutputFile file : result.getOutputFilesList()) {
        Path path = execRoot.getRelative(file.getPath());
        FileSystemUtils.createDirectoryAndParents(path.getParentDirectory());
        Digest digest = file.getDigest();
        if (digest.getSizeBytes() == 0) {
          // Handle empty file locally.
          FileSystemUtils.writeContent(path, new byte[0]);
        } else {
          if (!file.getContent().isEmpty()) {
            try (OutputStream stream = path.getOutputStream()) {
              file.getContent().writeTo(stream);
            }
          } else {
            retrier.execute(
                () -> {
                  try (OutputStream stream = path.getOutputStream()) {
                    readBlob(digest, stream);
                  }
                  return null;
                });
            Digest receivedDigest = Digests.computeDigest(path);
            if (!receivedDigest.equals(digest)) {
              throw new IOException(
                  "Digest does not match " + receivedDigest + " != " + digest);
            }
          }
        }
        path.setExecutable(file.getIsExecutable());
      }
      if (!result.getOutputDirectoriesList().isEmpty()) {
        throw new UnsupportedOperationException();
      }
      // TODO(ulfjack): use same code as above also for stdout / stderr if applicable.
      downloadOutErr(result, outErr);
    } catch (IOException downloadException) {
      try {
        // Delete any (partially) downloaded output files, since any subsequent local execution
        // of this action may expect none of the output files to exist.
        for (OutputFile file : result.getOutputFilesList()) {
          execRoot.getRelative(file.getPath()).delete();
        }
        outErr.getOutputPath().delete();
        outErr.getErrorPath().delete();
      } catch (IOException e) {
        // If deleting of output files failed, we abort the build with a decent error message as
        // any subsequent local execution failure would likely be incomprehensible.

        // We don't propagate the downloadException, as this is a recoverable error and the cause
        // of the build failure is really that we couldn't delete output files.
        throw new EnvironmentalExecException("Failed to delete output files after incomplete "
            + "download. Cannot continue with local execution.", e, true);
      }
      throw downloadException;
    }
  }

  private void downloadOutErr(ActionResult result, FileOutErr outErr)
      throws IOException, InterruptedException {
    if (!result.getStdoutRaw().isEmpty()) {
      result.getStdoutRaw().writeTo(outErr.getOutputStream());
      outErr.getOutputStream().flush();
    } else if (result.hasStdoutDigest()) {
      byte[] stdoutBytes = downloadBlob(result.getStdoutDigest());
      outErr.getOutputStream().write(stdoutBytes);
      outErr.getOutputStream().flush();
    }
    if (!result.getStderrRaw().isEmpty()) {
      result.getStderrRaw().writeTo(outErr.getErrorStream());
      outErr.getErrorStream().flush();
    } else if (result.hasStderrDigest()) {
      byte[] stderrBytes = downloadBlob(result.getStderrDigest());
      outErr.getErrorStream().write(stderrBytes);
      outErr.getErrorStream().flush();
    }
  }

  /**
   * This method can throw {@link StatusRuntimeException}, but the RemoteCache interface does not
   * allow throwing such an exception. Any caller must make sure to catch the
   * {@link StatusRuntimeException}. Note that the retrier implicitly catches it, so if this is used
   * in the context of {@link Retrier#execute}, that's perfectly safe.
   *
   * <p>This method also converts any NOT_FOUND code returned from the server into a
   * {@link CacheNotFoundException}. TODO(olaola): this is not enough. NOT_FOUND can also be raised
   * by execute, in which case the server should return the missing digest in the Status.details
   * field. This should be part of the API.
   */
  private void readBlob(Digest digest, OutputStream stream)
      throws IOException, StatusRuntimeException {
    String resourceName = "";
    if (!options.remoteInstanceName.isEmpty()) {
      resourceName += options.remoteInstanceName + "/";
    }
    resourceName += "blobs/" + digest.getHash() + "/" + digest.getSizeBytes();
    try {
      Iterator<ReadResponse> replies = bsBlockingStub()
          .read(ReadRequest.newBuilder().setResourceName(resourceName).build());
      while (replies.hasNext()) {
        replies.next().getData().writeTo(stream);
      }
    } catch (StatusRuntimeException e) {
      if (e.getStatus().getCode() == Status.Code.NOT_FOUND) {
        throw new CacheNotFoundException(digest);
      }
      throw e;
    }
  }

  @Override
  public void upload(
      ActionKey actionKey,
      Path execRoot,
      Collection<Path> files,
      FileOutErr outErr,
      boolean uploadAction)
      throws IOException, InterruptedException {
    ActionResult.Builder result = ActionResult.newBuilder();
    upload(execRoot, files, outErr, result);
    if (!uploadAction) {
      return;
    }
    try {
      retrier.execute(
          () ->
              acBlockingStub()
                  .updateActionResult(
                      UpdateActionResultRequest.newBuilder()
                          .setInstanceName(options.remoteInstanceName)
                          .setActionDigest(actionKey.getDigest())
                          .setActionResult(result)
                          .build()));
    } catch (RetryException e) {
      if (e.causedByStatusCode(Status.Code.UNIMPLEMENTED)) {
        // Silently return without upload.
        return;
      }
      throw e;
    }
  }

  void upload(Path execRoot, Collection<Path> files, FileOutErr outErr, ActionResult.Builder result)
      throws IOException, InterruptedException {
    Map<Digest, Path> digestToFile = new HashMap<>();
    for (Path file : files) {
      if (!file.exists()) {
        // We ignore requested results that have not been generated by the action.
        continue;
      }
      if (file.isDirectory()) {
        // TODO(olaola): to implement this for a directory, will need to create or pass a
        // TreeNodeRepository to call uploadTree.
        throw new UnsupportedOperationException("Storing a directory is not yet supported.");
      }

      Digest digest = Digests.computeDigest(file);
      // TODO(olaola): inline small results here.
      result
          .addOutputFilesBuilder()
          .setPath(file.relativeTo(execRoot).getPathString())
          .setDigest(digest)
          .setIsExecutable(file.isExecutable());
      digestToFile.put(digest, file);
    }

    ImmutableSet<Digest> digestsToUpload = getMissingDigests(digestToFile.keySet());
    List<Chunker> filesToUpload = new ArrayList<>();
    for (Digest digest : digestsToUpload) {
      Path file = digestToFile.get(digest);
      if (file == null) {
        String message = "FindMissingBlobs call returned an unknown digest: " + digest;
        throw new IOException(message);
      }
      filesToUpload.add(new Chunker(file));
    }
    if (!filesToUpload.isEmpty()) {
      uploader.uploadBlobs(filesToUpload);
    }

    // TODO(olaola): inline small stdout/stderr here.
    if (outErr.getErrorPath().exists()) {
      Digest stderr = uploadFileContents(outErr.getErrorPath());
      result.setStderrDigest(stderr);
    }
    if (outErr.getOutputPath().exists()) {
      Digest stdout = uploadFileContents(outErr.getOutputPath());
      result.setStdoutDigest(stdout);
    }
  }

  /**
   * Put the file contents cache if it is not already in it. No-op if the file is already stored in
   * cache. The given path must be a full absolute path.
   *
   * @return The key for fetching the file contents blob from cache.
   */
  private Digest uploadFileContents(Path file) throws IOException, InterruptedException {
    Digest digest = Digests.computeDigest(file);
    ImmutableSet<Digest> missing = getMissingDigests(ImmutableList.of(digest));
    if (!missing.isEmpty()) {
      uploader.uploadBlob(new Chunker(file));
    }
    return digest;
  }

  /**
   * Put the file contents cache if it is not already in it. No-op if the file is already stored in
   * cache. The given path must be a full absolute path.
   *
   * @return The key for fetching the file contents blob from cache.
   */
  Digest uploadFileContents(ActionInput input, Path execRoot, MetadataProvider inputCache)
      throws IOException, InterruptedException {
    Digest digest = Digests.getDigestFromInputCache(input, inputCache);
    ImmutableSet<Digest> missing = getMissingDigests(ImmutableList.of(digest));
    if (!missing.isEmpty()) {
      uploader.uploadBlob(new Chunker(input, inputCache, execRoot));
    }
    return digest;
  }

  Digest uploadBlob(byte[] blob) throws IOException, InterruptedException {
    Digest digest = Digests.computeDigest(blob);
    ImmutableSet<Digest> missing = getMissingDigests(ImmutableList.of(digest));
    if (!missing.isEmpty()) {
      uploader.uploadBlob(new Chunker(blob));
    }
    return digest;
  }

  byte[] downloadBlob(Digest digest)
      throws IOException, InterruptedException {
    if (digest.getSizeBytes() == 0) {
      return new byte[0];
    }
    return retrier.execute(
        () -> {
          ByteArrayOutputStream stream = new ByteArrayOutputStream((int) digest.getSizeBytes());
          readBlob(digest, stream);
          return stream.toByteArray();
        });
  }

  // Execution Cache API

  @Override
  public ActionResult getCachedActionResult(ActionKey actionKey)
      throws IOException, InterruptedException {
    try {
      return retrier.execute(
          () ->
              acBlockingStub()
                  .getActionResult(
                      GetActionResultRequest.newBuilder()
                          .setInstanceName(options.remoteInstanceName)
                          .setActionDigest(actionKey.getDigest())
                          .build()));
    } catch (RetryException e) {
      if (e.causedByStatusCode(Status.Code.NOT_FOUND)) {
        // Return null to indicate that it was a cache miss.
        return null;
      }
      throw e;
    }
  }
}

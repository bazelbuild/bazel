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
import com.google.bytestream.ByteStreamGrpc.ByteStreamStub;
import com.google.bytestream.ByteStreamProto.ReadRequest;
import com.google.bytestream.ByteStreamProto.ReadResponse;
import com.google.bytestream.ByteStreamProto.WriteRequest;
import com.google.bytestream.ByteStreamProto.WriteResponse;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.remote.Digests.ActionKey;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.remoteexecution.v1test.ActionCacheGrpc;
import com.google.devtools.remoteexecution.v1test.ActionCacheGrpc.ActionCacheBlockingStub;
import com.google.devtools.remoteexecution.v1test.ActionResult;
import com.google.devtools.remoteexecution.v1test.BatchUpdateBlobsRequest;
import com.google.devtools.remoteexecution.v1test.BatchUpdateBlobsResponse;
import com.google.devtools.remoteexecution.v1test.Command;
import com.google.devtools.remoteexecution.v1test.ContentAddressableStorageGrpc;
import com.google.devtools.remoteexecution.v1test.ContentAddressableStorageGrpc.ContentAddressableStorageBlockingStub;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.devtools.remoteexecution.v1test.Directory;
import com.google.devtools.remoteexecution.v1test.FindMissingBlobsRequest;
import com.google.devtools.remoteexecution.v1test.FindMissingBlobsResponse;
import com.google.devtools.remoteexecution.v1test.GetActionResultRequest;
import com.google.devtools.remoteexecution.v1test.OutputDirectory;
import com.google.devtools.remoteexecution.v1test.OutputFile;
import com.google.devtools.remoteexecution.v1test.UpdateActionResultRequest;
import com.google.protobuf.ByteString;
import io.grpc.Channel;
import io.grpc.Status;
import io.grpc.StatusException;
import io.grpc.StatusRuntimeException;
import io.grpc.protobuf.StatusProto;
import io.grpc.stub.StreamObserver;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/** A RemoteActionCache implementation that uses gRPC calls to a remote cache server. */
@ThreadSafe
public class GrpcActionCache implements RemoteActionCache {
  private final RemoteOptions options;
  private final ChannelOptions channelOptions;
  private final Channel channel;
  private final Retrier retrier;
  // All gRPC stubs are reused.
  private final Supplier<ContentAddressableStorageBlockingStub> casBlockingStub;
  private final Supplier<ByteStreamBlockingStub> bsBlockingStub;
  private final Supplier<ByteStreamStub> bsStub;
  private final Supplier<ActionCacheBlockingStub> acBlockingStub;

  @VisibleForTesting
  public GrpcActionCache(Channel channel, ChannelOptions channelOptions, RemoteOptions options) {
    this.options = options;
    this.channelOptions = channelOptions;
    this.channel = channel;
    this.retrier = new Retrier(options);
    casBlockingStub =
        Suppliers.memoize(
            () ->
                ContentAddressableStorageGrpc.newBlockingStub(channel)
                    .withCallCredentials(channelOptions.getCallCredentials())
                    .withDeadlineAfter(options.remoteTimeout, TimeUnit.SECONDS));
    bsBlockingStub =
        Suppliers.memoize(
            () ->
                ByteStreamGrpc.newBlockingStub(channel)
                    .withCallCredentials(channelOptions.getCallCredentials())
                    .withDeadlineAfter(options.remoteTimeout, TimeUnit.SECONDS));
    bsStub =
        Suppliers.memoize(
            () ->
                ByteStreamGrpc.newStub(channel)
                    .withCallCredentials(channelOptions.getCallCredentials())
                    .withDeadlineAfter(options.remoteTimeout, TimeUnit.SECONDS));
    acBlockingStub =
        Suppliers.memoize(
            () ->
                ActionCacheGrpc.newBlockingStub(channel)
                    .withCallCredentials(channelOptions.getCallCredentials())
                    .withDeadlineAfter(options.remoteTimeout, TimeUnit.SECONDS));
  }

  @Override
  public void close() {}

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
        retrier.execute(() -> casBlockingStub.get().findMissingBlobs(request.build()));
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
    // TODO(olaola): avoid querying all the digests, only ask for novel subtrees.
    ImmutableSet<Digest> missingDigests = getMissingDigests(repository.getAllDigests(root));

    // Only upload data that was missing from the cache.
    ArrayList<ActionInput> actionInputs = new ArrayList<>();
    ArrayList<Directory> treeNodes = new ArrayList<>();
    repository.getDataFromDigests(missingDigests, actionInputs, treeNodes);

    if (!treeNodes.isEmpty()) {
      // TODO(olaola): split this into multiple requests if total size is > 10MB.
      BatchUpdateBlobsRequest.Builder treeBlobRequest =
          BatchUpdateBlobsRequest.newBuilder().setInstanceName(options.remoteInstanceName);
      for (Directory d : treeNodes) {
        byte[] data = d.toByteArray();
        treeBlobRequest
            .addRequestsBuilder()
            .setContentDigest(Digests.computeDigest(data))
            .setData(ByteString.copyFrom(data));
      }
      retrier.execute(
          () -> {
            BatchUpdateBlobsResponse response =
                casBlockingStub.get().batchUpdateBlobs(treeBlobRequest.build());
            for (BatchUpdateBlobsResponse.Response r : response.getResponsesList()) {
              if (!Status.fromCodeValue(r.getStatus().getCode()).isOk()) {
                throw StatusProto.toStatusRuntimeException(r.getStatus());
              }
            }
            return null;
          });
    }
    uploadBlob(command.toByteArray());
    if (!actionInputs.isEmpty()) {
      uploadChunks(
          actionInputs.size(),
          new Chunker.Builder()
              .addAllInputs(actionInputs, repository.getInputFileCache(), execRoot)
              .onlyUseDigests(missingDigests));
    }
  }

  /**
   * Download the entire tree data rooted by the given digest and write it into the given location.
   */
  @SuppressWarnings("unused")
  private void downloadTree(Digest rootDigest, Path rootLocation) {
    throw new UnsupportedOperationException();
  }

  /**
   * Download all results of a remotely executed action locally. TODO(olaola): will need to amend to
   * include the {@link com.google.devtools.build.lib.remote.TreeNodeRepository} for updating.
   */
  @Override
  public void download(ActionResult result, Path execRoot, FileOutErr outErr)
      throws IOException, InterruptedException, CacheNotFoundException {
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
                  Iterator<ReadResponse> replies = readBlob(digest);
                  while (replies.hasNext()) {
                    replies.next().getData().writeTo(stream);
                  }
                  return null;
                }
              });
        }
      }
      path.setExecutable(file.getIsExecutable());
    }
    for (OutputDirectory directory : result.getOutputDirectoriesList()) {
      downloadTree(directory.getDigest(), execRoot.getRelative(directory.getPath()));
    }
    // TODO(ulfjack): use same code as above also for stdout / stderr if applicable.
    downloadOutErr(result, outErr);
  }

  private void downloadOutErr(ActionResult result, FileOutErr outErr)
      throws IOException, InterruptedException, CacheNotFoundException {
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

  private Iterator<ReadResponse> readBlob(Digest digest) throws CacheNotFoundException {
    String resourceName = "";
    if (!options.remoteInstanceName.isEmpty()) {
      resourceName += options.remoteInstanceName + "/";
    }
    resourceName += "blobs/" + digest.getHash() + "/" + digest.getSizeBytes();
    try {
      return bsBlockingStub
          .get()
          .read(ReadRequest.newBuilder().setResourceName(resourceName).build());
    } catch (StatusRuntimeException e) {
      if (e.getStatus().getCode() == Status.Code.NOT_FOUND) {
        throw new CacheNotFoundException(digest);
      }
      throw e;
    }
  }

  @Override
  public void upload(ActionKey actionKey, Path execRoot, Collection<Path> files, FileOutErr outErr)
      throws IOException, InterruptedException {
    ActionResult.Builder result = ActionResult.newBuilder();
    upload(execRoot, files, outErr, result);
    try {
      retrier.execute(
          () ->
              acBlockingStub
                  .get()
                  .updateActionResult(
                      UpdateActionResultRequest.newBuilder()
                          .setInstanceName(options.remoteInstanceName)
                          .setActionDigest(actionKey.getDigest())
                          .setActionResult(result)
                          .build()));
    } catch (StatusRuntimeException e) {
      if (e.getStatus().getCode() != Status.Code.UNIMPLEMENTED) {
        throw e;
      }
    }
  }

  void upload(Path execRoot, Collection<Path> files, FileOutErr outErr, ActionResult.Builder result)
      throws IOException, InterruptedException {
    ArrayList<Digest> digests = new ArrayList<>();
    Chunker.Builder b = new Chunker.Builder();
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
      digests.add(Digests.computeDigest(file));
      b.addInput(file);
    }
    ImmutableSet<Digest> missing = getMissingDigests(digests);
    if (!missing.isEmpty()) {
      uploadChunks(missing.size(), b.onlyUseDigests(missing));
    }
    int index = 0;
    for (Path file : files) {
      // Add to protobuf.
      // TODO(olaola): inline small results here.
      result
          .addOutputFilesBuilder()
          .setPath(file.relativeTo(execRoot).getPathString())
          .setDigest(digests.get(index++))
          .setIsExecutable(file.isExecutable());
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
      uploadChunks(1, new Chunker.Builder().addInput(file));
    }
    return digest;
  }

  /**
   * Put the file contents cache if it is not already in it. No-op if the file is already stored in
   * cache. The given path must be a full absolute path.
   *
   * @return The key for fetching the file contents blob from cache.
   */
  Digest uploadFileContents(ActionInput input, Path execRoot, ActionInputFileCache inputCache)
      throws IOException, InterruptedException {
    Digest digest = Digests.getDigestFromInputCache(input, inputCache);
    ImmutableSet<Digest> missing = getMissingDigests(ImmutableList.of(digest));
    if (!missing.isEmpty()) {
      uploadChunks(1, new Chunker.Builder().addInput(input, inputCache, execRoot));
    }
    return digest;
  }

  private void uploadChunks(int numItems, Chunker.Builder chunkerBuilder)
      throws InterruptedException, IOException {
    String resourceName = "";
    if (!options.remoteInstanceName.isEmpty()) {
      resourceName += options.remoteInstanceName + "/";
    }
    Retrier.Backoff backoff = retrier.newBackoff();
    Chunker chunker = chunkerBuilder.build();
    while (true) { // Retry until either uploaded everything or raised an exception.
      CountDownLatch finishLatch = new CountDownLatch(numItems);
      AtomicReference<IOException> crashException = new AtomicReference<>(null);
      List<Status> errors = Collections.synchronizedList(new ArrayList<Status>());
      Set<Digest> failedDigests = Collections.synchronizedSet(new HashSet<Digest>());
      StreamObserver<WriteRequest> requestObserver = null;
      while (chunker.hasNext()) {
        Chunker.Chunk chunk = chunker.next();
        Digest digest = chunk.getDigest();
        long offset = chunk.getOffset();
        WriteRequest.Builder request = WriteRequest.newBuilder();
        if (offset == 0) { // Beginning of new upload.
          numItems--;
          request.setResourceName(
              String.format(
                  "%s/uploads/%s/blobs/%s/%d",
                  resourceName, UUID.randomUUID(), digest.getHash(), digest.getSizeBytes()));
          // The batches execute simultaneously.
          requestObserver =
              bsStub
                  .get()
                  .write(
                      new StreamObserver<WriteResponse>() {
                        private long bytesLeft = digest.getSizeBytes();

                        @Override
                        public void onNext(WriteResponse reply) {
                          bytesLeft -= reply.getCommittedSize();
                        }

                        @Override
                        public void onError(Throwable t) {
                          // In theory, this can be any error, even though it's supposed to usually
                          // be only StatusException or StatusRuntimeException. We have to check
                          // for other errors, in order to not accidentally retry them!
                          if (!(t instanceof StatusRuntimeException
                              || t instanceof StatusException)) {
                            crashException.compareAndSet(null, new IOException(t));
                          }

                          failedDigests.add(digest);
                          errors.add(Status.fromThrowable(t));
                          finishLatch.countDown();
                        }

                        @Override
                        public void onCompleted() {
                          // This can actually happen even if we did not send all the bytes,
                          // if the server has and is able to reuse parts of the uploaded blob.
                          finishLatch.countDown();
                        }
                      });
        }
        byte[] data = chunk.getData();
        boolean finishWrite = offset + data.length == digest.getSizeBytes();
        request
            .setData(ByteString.copyFrom(data))
            .setWriteOffset(offset)
            .setFinishWrite(finishWrite);
        requestObserver.onNext(request.build());
        if (finishWrite) {
          requestObserver.onCompleted();
        }
        if (finishLatch.getCount() <= numItems) {
          // Current RPC errored before we finished sending.
          if (!finishWrite) {
            chunker.advanceInput();
          }
        }
      }
      finishLatch.await(options.remoteTimeout, TimeUnit.SECONDS);
      if (crashException.get() != null) {
        throw crashException.get(); // Re-throw the exception that is supposed to never happen.
      }
      if (failedDigests.isEmpty()) {
        return; // Successfully sent everything.
      }
      retrier.onFailures(backoff, errors); // This will throw when out of retries.
      // We don't have to synchronize on failedDigests now, because after finishLatch.await we're
      // back to single threaded execution.
      chunker = chunkerBuilder.onlyUseDigests(failedDigests).build();
      numItems = failedDigests.size();
    }
  }

  Digest uploadBlob(byte[] blob) throws IOException, InterruptedException {
    Digest digest = Digests.computeDigest(blob);
    ImmutableSet<Digest> missing = getMissingDigests(ImmutableList.of(digest));
    try {
      if (!missing.isEmpty()) {
        uploadChunks(1, new Chunker.Builder().addInput(blob));
      }
      return digest;
    } catch (IOException e) {
      // This will never happen.
      throw new RuntimeException();
    }
  }

  byte[] downloadBlob(Digest digest)
      throws IOException, InterruptedException, CacheNotFoundException {
    if (digest.getSizeBytes() == 0) {
      return new byte[0];
    }
    byte[] result = new byte[(int) digest.getSizeBytes()];
    retrier.execute(
        () -> {
          Iterator<ReadResponse> replies = readBlob(digest);
          int offset = 0;
          while (replies.hasNext()) {
            ByteString data = replies.next().getData();
            data.copyTo(result, offset);
            offset += data.size();
          }
          Preconditions.checkState(digest.getSizeBytes() == offset);
          return null;
        });
    return result;
  }

  // Execution Cache API

  /** Returns a cached result for a given Action digest, or null if not found in cache. */
  @Override
  public ActionResult getCachedActionResult(ActionKey actionKey)
      throws IOException, InterruptedException {
    try {
      return retrier.execute(
          () ->
              acBlockingStub
                  .get()
                  .getActionResult(
                      GetActionResultRequest.newBuilder()
                          .setInstanceName(options.remoteInstanceName)
                          .setActionDigest(actionKey.getDigest())
                          .build()));
    } catch (RetryException e) {
      if (e.causedByStatusCode(Status.Code.NOT_FOUND)) {
        return null;
      }
      throw e;
    }
  }
}

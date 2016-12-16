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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterators;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.remote.CasServiceGrpc.CasServiceBlockingStub;
import com.google.devtools.build.lib.remote.CasServiceGrpc.CasServiceStub;
import com.google.devtools.build.lib.remote.ContentDigests.ActionKey;
import com.google.devtools.build.lib.remote.ExecutionCacheServiceGrpc.ExecutionCacheServiceBlockingStub;
import com.google.devtools.build.lib.remote.RemoteProtocol.ActionResult;
import com.google.devtools.build.lib.remote.RemoteProtocol.BlobChunk;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadBlobRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasLookupRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasStatus;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadBlobReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadBlobRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadTreeMetadataReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadTreeMetadataRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ContentDigest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheSetReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheSetRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheStatus;
import com.google.devtools.build.lib.remote.RemoteProtocol.FileMetadata;
import com.google.devtools.build.lib.remote.RemoteProtocol.FileNode;
import com.google.devtools.build.lib.remote.RemoteProtocol.Output;
import com.google.devtools.build.lib.remote.RemoteProtocol.Output.ContentCase;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import io.grpc.ManagedChannel;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.StreamObserver;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/** A RemoteActionCache implementation that uses gRPC calls to a remote cache server. */
@ThreadSafe
public final class GrpcActionCache implements RemoteActionCache {

  /** Channel over which to send gRPC CAS queries. */
  private final ManagedChannel channel;

  private final RemoteOptions options;

  private static final int MAX_MEMORY_KBYTES = 512 * 1024;

  /** Reads from multiple sequential inputs and chunks the data into BlobChunks. */
  static interface BlobChunkIterator {
    boolean hasNext();

    BlobChunk next() throws IOException; // IOException can be a result of file read.
  }

  final class BlobChunkInlineIterator implements BlobChunkIterator {
    private final Iterator<byte[]> blobIterator;
    private final Set<ContentDigest> digests;
    private int offset;
    private ContentDigest digest;
    private byte[] currentBlob;

    public BlobChunkInlineIterator(Set<ContentDigest> digests, Iterator<byte[]> blobIterator) {
      this.digests = digests;
      this.blobIterator = blobIterator;
      advanceInput();
    }

    public BlobChunkInlineIterator(byte[] blob) {
      blobIterator = null;
      offset = 0;
      currentBlob = blob;
      digest = ContentDigests.computeDigest(currentBlob);
      digests = null;
    }

    private void advanceInput() {
      offset = 0;
      do {
        if (blobIterator != null && blobIterator.hasNext()) {
          currentBlob = blobIterator.next();
          digest = ContentDigests.computeDigest(currentBlob);
        } else {
          currentBlob = null;
          digest = null;
        }
      } while (digest != null && !digests.contains(digest));
    }

    @Override
    public boolean hasNext() {
      return currentBlob != null;
    }

    @Override
    public BlobChunk next() throws IOException {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      BlobChunk.Builder chunk = BlobChunk.newBuilder();
      if (offset == 0) {
        chunk.setDigest(digest);
      } else {
        chunk.setOffset(offset);
      }
      int size = Math.min(currentBlob.length - offset, options.grpcMaxChunkSizeBytes);
      if (size > 0) {
        chunk.setData(ByteString.copyFrom(currentBlob, offset, size));
        offset += size;
      }
      if (offset >= currentBlob.length) {
        advanceInput();
      }
      return chunk.build();
    }
  }

  final class BlobChunkFileIterator implements BlobChunkIterator {
    private final Iterator<Path> fileIterator;
    private InputStream currentStream;
    private final Set<ContentDigest> digests;
    private ContentDigest digest;
    private long bytesLeft;

    public BlobChunkFileIterator(Set<ContentDigest> digests, Iterator<Path> fileIterator)
        throws IOException {
      this.digests = digests;
      this.fileIterator = fileIterator;
      advanceInput();
    }

    public BlobChunkFileIterator(Path file) throws IOException {
      fileIterator = Iterators.singletonIterator(file);
      digests = ImmutableSet.of(ContentDigests.computeDigest(file));
      advanceInput();
    }

    private void advanceInput() throws IOException {
      do {
        if (fileIterator != null && fileIterator.hasNext()) {
          Path file = fileIterator.next();
          digest = ContentDigests.computeDigest(file);
          currentStream = file.getInputStream();
          bytesLeft = digest.getSizeBytes();
        } else {
          digest = null;
          currentStream = null;
          bytesLeft = 0;
        }
      } while (digest != null && !digests.contains(digest));
    }

    @Override
    public boolean hasNext() {
      return currentStream != null;
    }

    @Override
    public BlobChunk next() throws IOException {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      BlobChunk.Builder chunk = BlobChunk.newBuilder();
      long offset = digest.getSizeBytes() - bytesLeft;
      if (offset == 0) {
        chunk.setDigest(digest);
      } else {
        chunk.setOffset(offset);
      }
      if (bytesLeft > 0) {
        byte[] blob = new byte[(int) Math.min(bytesLeft, (long) options.grpcMaxChunkSizeBytes)];
        currentStream.read(blob);
        chunk.setData(ByteString.copyFrom(blob));
        bytesLeft -= blob.length;
      }
      if (bytesLeft == 0) {
        currentStream.close();
        advanceInput();
      }
      return chunk.build();
    }
  }

  @VisibleForTesting
  public GrpcActionCache(ManagedChannel channel, RemoteOptions options) {
    this.options = options;
    this.channel = channel;
  }

  public GrpcActionCache(RemoteOptions options) throws InvalidConfigurationException {
    this(RemoteUtils.createChannel(options.remoteCache), options);
  }

  public static boolean isRemoteCacheOptions(RemoteOptions options) {
    return options.remoteCache != null;
  }

  private CasServiceBlockingStub getBlockingStub() {
    return CasServiceGrpc.newBlockingStub(channel)
        .withDeadlineAfter(options.grpcTimeoutSeconds, TimeUnit.SECONDS);
  }

  private CasServiceStub getStub() {
    return CasServiceGrpc.newStub(channel).withDeadlineAfter(
        options.grpcTimeoutSeconds, TimeUnit.SECONDS);
  }

  private ImmutableSet<ContentDigest> getMissingDigests(Iterable<ContentDigest> digests) {
    CasLookupRequest.Builder request = CasLookupRequest.newBuilder().addAllDigest(digests);
    if (request.getDigestCount() == 0) {
      return ImmutableSet.of();
    }
    CasStatus status = getBlockingStub().lookup(request.build()).getStatus();
    if (!status.getSucceeded() && status.getError() != CasStatus.ErrorCode.MISSING_DIGEST) {
      // TODO(olaola): here and below, add basic retry logic on transient errors!
      throw new RuntimeException(status.getErrorDetail());
    }
    return ImmutableSet.copyOf(status.getMissingDigestList());
  }

  /**
   * Upload enough of the tree metadata and data into remote cache so that the entire tree can be
   * reassembled remotely using the root digest.
   */
  @Override
  public void uploadTree(TreeNodeRepository repository, Path execRoot, TreeNode root)
      throws IOException, InterruptedException {
    repository.computeMerkleDigests(root);
    // TODO(olaola): avoid querying all the digests, only ask for novel subtrees.
    ImmutableSet<ContentDigest> missingDigests = getMissingDigests(repository.getAllDigests(root));

    // Only upload data that was missing from the cache.
    ArrayList<ActionInput> actionInputs = new ArrayList<>();
    ArrayList<FileNode> treeNodes = new ArrayList<>();
    repository.getDataFromDigests(missingDigests, actionInputs, treeNodes);

    if (!treeNodes.isEmpty()) {
      CasUploadTreeMetadataRequest.Builder metaRequest =
          CasUploadTreeMetadataRequest.newBuilder().addAllTreeNode(treeNodes);
      CasUploadTreeMetadataReply reply = getBlockingStub().uploadTreeMetadata(metaRequest.build());
      if (!reply.getStatus().getSucceeded()) {
        throw new RuntimeException(reply.getStatus().getErrorDetail());
      }
    }
    if (!actionInputs.isEmpty()) {
      ArrayList<Path> paths = new ArrayList<>();
      for (ActionInput actionInput : actionInputs) {
        paths.add(execRoot.getRelative(actionInput.getExecPathString()));
      }
      uploadChunks(paths.size(), new BlobChunkFileIterator(missingDigests, paths.iterator()));
    }
  }

  /**
   * Download the entire tree data rooted by the given digest and write it into the given location.
   */
  @Override
  public void downloadTree(ContentDigest rootDigest, Path rootLocation)
      throws IOException, CacheNotFoundException {
    throw new UnsupportedOperationException();
  }

  private void handleDownloadStatus(CasStatus status) throws CacheNotFoundException {
    if (!status.getSucceeded()) {
      if (status.getError() == CasStatus.ErrorCode.MISSING_DIGEST) {
        throw new CacheNotFoundException(status.getMissingDigest(0));
      }
      // TODO(olaola): deal with other statuses better.
      throw new RuntimeException(status.getErrorDetail());
    }
  }

  /**
   * Download all results of a remotely executed action locally. TODO(olaola): will need to amend to
   * include the {@link com.google.devtools.build.lib.remote.TreeNodeRepository} for updating.
   */
  @Override
  public void downloadAllResults(ActionResult result, Path execRoot)
      throws IOException, CacheNotFoundException {
    // Send all the file requests in a single synchronous batch.
    // TODO(olaola): profile to maybe replace with separate concurrent requests.
    CasDownloadBlobRequest.Builder request = CasDownloadBlobRequest.newBuilder();
    ArrayList<Output> fileOutputs = new ArrayList<>();
    for (Output output : result.getOutputList()) {
      Path path = execRoot.getRelative(output.getPath());
      if (output.getContentCase() == ContentCase.FILE_METADATA) {
        ContentDigest digest = output.getFileMetadata().getDigest();
        if (digest.getSizeBytes() > 0) {
          request.addDigest(digest);
          fileOutputs.add(output);
        } else {
          // Handle empty file locally.
          FileSystemUtils.createDirectoryAndParents(path.getParentDirectory());
          FileSystemUtils.writeContent(path, new byte[0]);
        }
      } else {
        downloadTree(output.getDigest(), path);
      }
    }
    Iterator<CasDownloadReply> replies = getBlockingStub().downloadBlob(request.build());
    for (Output output : fileOutputs) {
      createFileFromStream(
          execRoot.getRelative(output.getPath()), output.getFileMetadata(), replies);
    }
  }

  private void createFileFromStream(
      Path path, FileMetadata fileMetadata, Iterator<CasDownloadReply> replies)
      throws IOException, CacheNotFoundException {
    Preconditions.checkArgument(replies.hasNext());
    CasDownloadReply reply = replies.next();
    if (reply.hasStatus()) {
      handleDownloadStatus(reply.getStatus());
    }
    BlobChunk chunk = reply.getData();
    ContentDigest digest = chunk.getDigest();
    Preconditions.checkArgument(digest.equals(fileMetadata.getDigest()));
    FileSystemUtils.createDirectoryAndParents(path.getParentDirectory());
    try (OutputStream stream = path.getOutputStream()) {
      ByteString data = chunk.getData();
      data.writeTo(stream);
      long bytesLeft = digest.getSizeBytes() - data.size();
      while (bytesLeft > 0) {
        Preconditions.checkArgument(replies.hasNext());
        reply = replies.next();
        if (reply.hasStatus()) {
          handleDownloadStatus(reply.getStatus());
        }
        chunk = reply.getData();
        data = chunk.getData();
        Preconditions.checkArgument(!chunk.hasDigest());
        Preconditions.checkArgument(chunk.getOffset() == digest.getSizeBytes() - bytesLeft);
        data.writeTo(stream);
        bytesLeft -= data.size();
      }
      path.setExecutable(fileMetadata.getExecutable());
    }
  }

  private byte[] getBlobFromStream(ContentDigest blobDigest, Iterator<CasDownloadReply> replies)
      throws CacheNotFoundException {
    Preconditions.checkArgument(replies.hasNext());
    CasDownloadReply reply = replies.next();
    if (reply.hasStatus()) {
      handleDownloadStatus(reply.getStatus());
    }
    BlobChunk chunk = reply.getData();
    ContentDigest digest = chunk.getDigest();
    Preconditions.checkArgument(digest.equals(blobDigest));
    // This is not enough, but better than nothing.
    Preconditions.checkArgument(digest.getSizeBytes() / 1000.0 < MAX_MEMORY_KBYTES);
    byte[] result = new byte[(int) digest.getSizeBytes()];
    ByteString data = chunk.getData();
    data.copyTo(result, 0);
    int offset = data.size();
    while (offset < result.length) {
      Preconditions.checkArgument(replies.hasNext());
      reply = replies.next();
      if (reply.hasStatus()) {
        handleDownloadStatus(reply.getStatus());
      }
      chunk = reply.getData();
      Preconditions.checkArgument(!chunk.hasDigest());
      Preconditions.checkArgument(chunk.getOffset() == offset);
      data = chunk.getData();
      data.copyTo(result, offset);
      offset += data.size();
    }
    return result;
  }

  /** Upload all results of a locally executed action to the cache. */
  @Override
  public void uploadAllResults(Path execRoot, Collection<Path> files, ActionResult.Builder result)
      throws IOException, InterruptedException {
    ArrayList<ContentDigest> digests = new ArrayList<>();
    for (Path file : files) {
      digests.add(ContentDigests.computeDigest(file));
    }
    ImmutableSet<ContentDigest> missing = getMissingDigests(digests);
    if (!missing.isEmpty()) {
      uploadChunks(missing.size(), new BlobChunkFileIterator(missing, files.iterator()));
    }
    int index = 0;
    for (Path file : files) {
      if (file.isDirectory()) {
        // TODO(olaola): to implement this for a directory, will need to create or pass a
        // TreeNodeRepository to call uploadTree.
        throw new UnsupportedOperationException("Storing a directory is not yet supported.");
      }
      // Add to protobuf.
      result
          .addOutputBuilder()
          .setPath(file.relativeTo(execRoot).getPathString())
          .getFileMetadataBuilder()
          .setDigest(digests.get(index++))
          .setExecutable(file.isExecutable());
    }
  }

  /**
   * Put the file contents cache if it is not already in it. No-op if the file is already stored in
   * cache. The given path must be a full absolute path. Note: this is horribly inefficient, need to
   * patch through an overload that uses an ActionInputFile cache to compute the digests!
   *
   * @return The key for fetching the file contents blob from cache.
   */
  @Override
  public ContentDigest uploadFileContents(Path file) throws IOException, InterruptedException {
    ContentDigest digest = ContentDigests.computeDigest(file);
    ImmutableSet<ContentDigest> missing = getMissingDigests(ImmutableList.of(digest));
    if (!missing.isEmpty()) {
      uploadChunks(1, new BlobChunkFileIterator(file));
    }
    return digest;
  }

  /**
   * Download a blob keyed by the given digest and write it to the specified path. Set the
   * executable parameter to the specified value.
   */
  @Override
  public void downloadFileContents(ContentDigest digest, Path dest, boolean executable)
      throws IOException, CacheNotFoundException {
    // Send all the file requests in a single synchronous batch.
    // TODO(olaola): profile to maybe replace with separate concurrent requests.
    CasDownloadBlobRequest.Builder request = CasDownloadBlobRequest.newBuilder().addDigest(digest);
    Iterator<CasDownloadReply> replies = getBlockingStub().downloadBlob(request.build());
    FileMetadata fileMetadata =
        FileMetadata.newBuilder().setDigest(digest).setExecutable(executable).build();
    createFileFromStream(dest, fileMetadata, replies);
  }

  static class UploadBlobReplyStreamObserver implements StreamObserver<CasUploadBlobReply> {
    private final CountDownLatch finishLatch;
    private final AtomicReference<RuntimeException> exception;

    public UploadBlobReplyStreamObserver(
        CountDownLatch finishLatch, AtomicReference<RuntimeException> exception) {
      this.finishLatch = finishLatch;
      this.exception = exception;
    }

    @Override
    public void onNext(CasUploadBlobReply reply) {
      if (!reply.getStatus().getSucceeded()) {
        // TODO(olaola): add basic retry logic on transient errors!
        this.exception.compareAndSet(
            null, new RuntimeException(reply.getStatus().getErrorDetail()));
      }
    }

    @Override
    public void onError(Throwable t) {
      this.exception.compareAndSet(null, new StatusRuntimeException(Status.fromThrowable(t)));
      finishLatch.countDown();
    }

    @Override
    public void onCompleted() {
      finishLatch.countDown();
    }
  }

  private void uploadChunks(int numItems, BlobChunkIterator blobs)
      throws InterruptedException, IOException {
    CountDownLatch finishLatch = new CountDownLatch(numItems); // Maximal number of batches.
    AtomicReference<RuntimeException> exception = new AtomicReference<>(null);
    UploadBlobReplyStreamObserver responseObserver = null;
    StreamObserver<CasUploadBlobRequest> requestObserver = null;
    int currentBatchBytes = 0;
    int batchedInputs = 0;
    int batches = 0;
    CasServiceStub stub = getStub();
    try {
      while (blobs.hasNext()) {
        BlobChunk chunk = blobs.next();
        if (chunk.hasDigest()) {
          // Determine whether to start next batch.
          final long batchSize = chunk.getDigest().getSizeBytes() + currentBatchBytes;
          if (batchedInputs % options.grpcMaxBatchInputs == 0
              || batchSize > options.grpcMaxBatchSizeBytes) {
            // The batches execute simultaneously.
            if (requestObserver != null) {
              batchedInputs = 0;
              currentBatchBytes = 0;
              requestObserver.onCompleted();
            }
            batches++;
            responseObserver = new UploadBlobReplyStreamObserver(finishLatch, exception);
            requestObserver = stub.uploadBlob(responseObserver);
          }
          batchedInputs++;
        }
        currentBatchBytes += chunk.getData().size();
        requestObserver.onNext(CasUploadBlobRequest.newBuilder().setData(chunk).build());
        if (finishLatch.getCount() == 0) {
          // RPC completed or errored before we finished sending.
          throw new RuntimeException(
              "gRPC terminated prematurely: "
                  + (exception.get() != null ? exception.get() : "unknown cause"));
        }
      }
    } catch (RuntimeException e) {
      // Cancel RPC
      if (requestObserver != null) {
        requestObserver.onError(e);
      }
      throw e;
    }
    if (requestObserver != null) {
      requestObserver.onCompleted(); // Finish last batch.
    }
    while (batches++ < numItems) {
      finishLatch.countDown(); // Non-sent batches.
    }
    finishLatch.await(options.grpcTimeoutSeconds, TimeUnit.SECONDS);
    if (exception.get() != null) {
      throw exception.get(); // Re-throw the first encountered exception.
    }
  }

  @Override
  public ImmutableList<ContentDigest> uploadBlobs(Iterable<byte[]> blobs)
      throws InterruptedException {
    ArrayList<ContentDigest> digests = new ArrayList<>();
    for (byte[] blob : blobs) {
      digests.add(ContentDigests.computeDigest(blob));
    }
    ImmutableSet<ContentDigest> missing = getMissingDigests(digests);
    try {
      if (!missing.isEmpty()) {
        uploadChunks(missing.size(), new BlobChunkInlineIterator(missing, blobs.iterator()));
      }
      return ImmutableList.copyOf(digests);
    } catch (IOException e) {
      // This will never happen.
      throw new RuntimeException(e);
    }
  }

  @Override
  public ContentDigest uploadBlob(byte[] blob) throws InterruptedException {
    ContentDigest digest = ContentDigests.computeDigest(blob);
    ImmutableSet<ContentDigest> missing = getMissingDigests(ImmutableList.of(digest));
    try {
      if (!missing.isEmpty()) {
        uploadChunks(1, new BlobChunkInlineIterator(blob));
      }
      return digest;
    } catch (IOException e) {
      // This will never happen.
      throw new RuntimeException();
    }
  }

  @Override
  public byte[] downloadBlob(ContentDigest digest) throws CacheNotFoundException {
    return downloadBlobs(ImmutableList.of(digest)).get(0);
  }

  @Override
  public ImmutableList<byte[]> downloadBlobs(Iterable<ContentDigest> digests)
      throws CacheNotFoundException {
    // Send all the file requests in a single synchronous batch.
    // TODO(olaola): profile to maybe replace with separate concurrent requests.
    CasDownloadBlobRequest.Builder request = CasDownloadBlobRequest.newBuilder();
    for (ContentDigest digest : digests) {
      if (digest.getSizeBytes() > 0) {
        request.addDigest(digest); // We handle empty blobs locally.
      }
    }
    Iterator<CasDownloadReply> replies = null;
    if (request.getDigestCount() > 0) {
      replies = getBlockingStub().downloadBlob(request.build());
    }
    ArrayList<byte[]> result = new ArrayList<>();
    for (ContentDigest digest : digests) {
      result.add(digest.getSizeBytes() == 0 ? new byte[0] : getBlobFromStream(digest, replies));
    }
    return ImmutableList.copyOf(result);
  }

  // Execution Cache API

  /** Returns a cached result for a given Action digest, or null if not found in cache. */
  @Override
  public ActionResult getCachedActionResult(ActionKey actionKey) {
    ExecutionCacheServiceBlockingStub stub =
        ExecutionCacheServiceGrpc.newBlockingStub(channel)
            .withDeadlineAfter(options.grpcTimeoutSeconds, TimeUnit.SECONDS);
    ExecutionCacheRequest request =
        ExecutionCacheRequest.newBuilder().setActionDigest(actionKey.getDigest()).build();
    ExecutionCacheReply reply = stub.getCachedResult(request);
    ExecutionCacheStatus status = reply.getStatus();
    if (!status.getSucceeded()
        && status.getError() != ExecutionCacheStatus.ErrorCode.MISSING_RESULT) {
      throw new RuntimeException(status.getErrorDetail());
    }
    return reply.hasResult() ? reply.getResult() : null;
  }

  /** Sets the given result as result of the given Action. */
  @Override
  public void setCachedActionResult(ActionKey actionKey, ActionResult result)
      throws InterruptedException {
    ExecutionCacheServiceBlockingStub stub =
        ExecutionCacheServiceGrpc.newBlockingStub(channel)
            .withDeadlineAfter(options.grpcTimeoutSeconds, TimeUnit.SECONDS);
    ExecutionCacheSetRequest request =
        ExecutionCacheSetRequest.newBuilder()
            .setActionDigest(actionKey.getDigest())
            .setResult(result)
            .build();
    ExecutionCacheSetReply reply = stub.setCachedResult(request);
    ExecutionCacheStatus status = reply.getStatus();
    if (!status.getSucceeded()
        && status.getError() != ExecutionCacheStatus.ErrorCode.UNSUPPORTED) {
      throw new RuntimeException(status.getErrorDetail());
    }
  }
}

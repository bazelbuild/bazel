// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.remote.worker;

import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;

import build.bazel.remote.execution.v2.BatchUpdateBlobsRequest;
import build.bazel.remote.execution.v2.BatchUpdateBlobsResponse;
import build.bazel.remote.execution.v2.ChunkingFunction;
import build.bazel.remote.execution.v2.ContentAddressableStorageGrpc.ContentAddressableStorageImplBase;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FindMissingBlobsRequest;
import build.bazel.remote.execution.v2.FindMissingBlobsResponse;
import build.bazel.remote.execution.v2.GetTreeRequest;
import build.bazel.remote.execution.v2.GetTreeResponse;
import build.bazel.remote.execution.v2.RequestMetadata;
import build.bazel.remote.execution.v2.SpliceBlobRequest;
import build.bazel.remote.execution.v2.SpliceBlobResponse;
import build.bazel.remote.execution.v2.SplitBlobRequest;
import build.bazel.remote.execution.v2.SplitBlobResponse;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.rpc.Code;
import io.grpc.stub.StreamObserver;
import com.google.devtools.build.lib.remote.util.DigestOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/** A basic implementation of a {@link ContentAddressableStorageImplBase} service. */
final class CasServer extends ContentAddressableStorageImplBase {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  static final long MAX_BATCH_SIZE_BYTES = 1024 * 1024 * 4;
  private final OnDiskBlobStoreCache cache;
  private final Map<Digest, List<Digest>> splicedBlobs = new ConcurrentHashMap<>();

  public CasServer(OnDiskBlobStoreCache cache) {
    this.cache = cache;
  }

  @Override
  public void findMissingBlobs(
      FindMissingBlobsRequest request, StreamObserver<FindMissingBlobsResponse> responseObserver) {
    FindMissingBlobsResponse.Builder response = FindMissingBlobsResponse.newBuilder();

    for (Digest digest : request.getBlobDigestsList()) {
      boolean exists = false;
      try {
        exists = cache.refresh(digest);
      } catch (IOException e) {
        responseObserver.onError(StatusUtils.internalError(e));
        return;
      }
      if (!exists) {
        response.addMissingBlobDigests(digest);
      }
    }

    responseObserver.onNext(response.build());
    responseObserver.onCompleted();
  }

  @Override
  public void batchUpdateBlobs(
      BatchUpdateBlobsRequest request, StreamObserver<BatchUpdateBlobsResponse> responseObserver) {
    RequestMetadata meta = TracingMetadataUtils.fromCurrentContext();
    RemoteActionExecutionContext context = RemoteActionExecutionContext.create(meta);

    BatchUpdateBlobsResponse.Builder batchResponse = BatchUpdateBlobsResponse.newBuilder();
    for (BatchUpdateBlobsRequest.Request r : request.getRequestsList()) {
      BatchUpdateBlobsResponse.Response.Builder resp = batchResponse.addResponsesBuilder();
      try {
        Digest digest = cache.getDigestUtil().compute(r.getData().toByteArray());
        getFromFuture(cache.uploadBlob(context, digest, r.getData()));
        if (!r.getDigest().equals(digest)) {
          String err =
              "Upload digest " + r.getDigest() + " did not match data digest: " + digest;
          resp.setStatus(StatusUtils.invalidArgumentStatus("digest", err));
          continue;
        }
        resp.getStatusBuilder().setCode(Code.OK.getNumber());
      } catch (Exception e) {
        resp.setStatus(StatusUtils.internalErrorStatus(e));
      }
    }
    responseObserver.onNext(batchResponse.build());
    responseObserver.onCompleted();
  }

  @Override
  public void getTree(GetTreeRequest request, StreamObserver<GetTreeResponse> responseObserver) {
    RequestMetadata meta = TracingMetadataUtils.fromCurrentContext();
    RemoteActionExecutionContext context = RemoteActionExecutionContext.create(meta);

    // Directories are returned in depth-first order.  We store all previously-traversed digests so
    // identical subtrees having the same digest will only be traversed and returned once.
    Set<Digest> seenDigests = new HashSet<>();
    Deque<Digest> pendingDigests = new ArrayDeque<>();
    seenDigests.add(request.getRootDigest());
    pendingDigests.push(request.getRootDigest());
    GetTreeResponse.Builder responseBuilder = GetTreeResponse.newBuilder();
    while (!pendingDigests.isEmpty()) {
      Digest digest = pendingDigests.pop();
      byte[] directoryBytes;
      try {
        directoryBytes = getFromFuture(cache.downloadBlob(context, digest));
      } catch (CacheNotFoundException e) {
        responseObserver.onError(StatusUtils.notFoundError(digest));
        return;
      } catch (InterruptedException e) {
        responseObserver.onError(StatusUtils.interruptedError(digest));
        return;
      } catch (Exception e) {
        logger.atWarning().withCause(e).log("Read request failed");
        responseObserver.onError(StatusUtils.internalError(e));
        return;
      }
      Directory directory;
      try {
        directory = Directory.parseFrom(directoryBytes);
      } catch (InvalidProtocolBufferException e) {
        logger.atWarning().withCause(e).log("Failed to parse directory in tree");
        responseObserver.onError(StatusUtils.internalError(e));
        return;
      }
      responseBuilder.addDirectories(directory);
      for (DirectoryNode directoryNode : directory.getDirectoriesList()) {
        if (seenDigests.add(directoryNode.getDigest())) {
          pendingDigests.push(directoryNode.getDigest());
        }
      }
    }
    responseObserver.onNext(responseBuilder.build());
    responseObserver.onCompleted();
  }

  /**
   * Returns the chunk digests for a blob that was previously stored via spliceBlob.
   * Clients use this to download large blobs in smaller pieces.
   */
  @Override
  public void splitBlob(
      SplitBlobRequest request, StreamObserver<SplitBlobResponse> responseObserver) {
    Digest blobDigest = request.getBlobDigest();

    List<Digest> chunkDigests = splicedBlobs.get(blobDigest);
    if (chunkDigests == null) {
      responseObserver.onError(StatusUtils.notFoundError(blobDigest));
      return;
    }
    responseObserver.onNext(
        SplitBlobResponse.newBuilder()
            .addAllChunkDigests(chunkDigests)
            .setChunkingFunction(ChunkingFunction.Value.FAST_CDC_2020)
            .build());
    responseObserver.onCompleted();
  }

  /**
   * Stores a mapping from a blob digest to the list of chunk digests that compose it.
   *
   * <p>All chunks must already exist in the CAS. The concatenated chunks are verified
   * to match the expected blob digest before storing the mapping.
   */
  @Override
  public void spliceBlob(
      SpliceBlobRequest request, StreamObserver<SpliceBlobResponse> responseObserver) {
    RequestMetadata meta = TracingMetadataUtils.fromCurrentContext();
    RemoteActionExecutionContext context = RemoteActionExecutionContext.create(meta);

    Digest blobDigest = request.getBlobDigest();
    List<Digest> chunkDigests = request.getChunkDigestsList();

    try {
      // Verify all chunks exist in the cache.
      for (Digest chunkDigest : chunkDigests) {
        if (!cache.refresh(chunkDigest)) {
          responseObserver.onError(StatusUtils.notFoundError(chunkDigest));
          return;
        }
      }

      DigestOutputStream digestOut =
          cache.getDigestUtil().newDigestOutputStream(OutputStream.nullOutputStream());
      for (Digest chunkDigest : chunkDigests) {
        byte[] chunkData = getFromFuture(cache.downloadBlob(context, chunkDigest));
        digestOut.write(chunkData);
      }
      Digest computedDigest = digestOut.digest();
      if (!computedDigest.equals(blobDigest)) {
        String err = "Splice digest " + blobDigest + " did not match computed digest: " + computedDigest;
        responseObserver.onError(StatusUtils.invalidArgumentError("blob_digest", err));
        return;
      }

      // Record the blob-to-chunks mapping for splitBlob lookups.
      splicedBlobs.put(blobDigest, new ArrayList<>(chunkDigests));

      responseObserver.onNext(
          SpliceBlobResponse.newBuilder().setBlobDigest(blobDigest).build());
      responseObserver.onCompleted();
    } catch (CacheNotFoundException e) {
      responseObserver.onError(StatusUtils.notFoundError(e.getMissingDigest()));
    } catch (InterruptedException e) {
      responseObserver.onError(StatusUtils.interruptedError(blobDigest));
    } catch (Exception e) {
      logger.atWarning().withCause(e).log("SpliceBlob request failed");
      responseObserver.onError(StatusUtils.internalError(e));
    }
  }
}

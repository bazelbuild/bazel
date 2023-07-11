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
import build.bazel.remote.execution.v2.ContentAddressableStorageGrpc.ContentAddressableStorageImplBase;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FindMissingBlobsRequest;
import build.bazel.remote.execution.v2.FindMissingBlobsResponse;
import build.bazel.remote.execution.v2.GetTreeRequest;
import build.bazel.remote.execution.v2.GetTreeResponse;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.rpc.Code;
import io.grpc.stub.StreamObserver;
import java.io.IOException;
import java.time.Instant;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Set;

/** A basic implementation of a {@link ContentAddressableStorageImplBase} service. */
final class CasServer extends ContentAddressableStorageImplBase {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  static final long MAX_BATCH_SIZE_BYTES = 1024 * 1024 * 4;
  private final OnDiskBlobStoreCache cache;

  public CasServer(OnDiskBlobStoreCache cache) {
    this.cache = cache;
  }

  @Override
  public void findMissingBlobs(
      FindMissingBlobsRequest request, StreamObserver<FindMissingBlobsResponse> responseObserver) {
    FindMissingBlobsResponse.Builder response = FindMissingBlobsResponse.newBuilder();

    for (Digest digest : request.getBlobDigestsList()) {
      if (!cache.containsKey(digest)) {
        response.addMissingBlobDigests(digest);
      } else {
        // Update mtime of referenced blobs to simulate lease extension.
        //
        // TODO(chiwang): Merge this into DiskCacheClient once we have implemented garbage
        // collection for it.
        var path = cache.getPath(digest);
        try {
          path.setLastModifiedTime(Instant.now().toEpochMilli());
        } catch (IOException e) {
          logger.atWarning().withCause(e).log("Failed to update mtime");
        }
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
}

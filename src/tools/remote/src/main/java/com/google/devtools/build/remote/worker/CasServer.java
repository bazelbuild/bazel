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

import build.bazel.remote.execution.v2.BatchUpdateBlobsRequest;
import build.bazel.remote.execution.v2.BatchUpdateBlobsResponse;
import build.bazel.remote.execution.v2.ContentAddressableStorageGrpc.ContentAddressableStorageImplBase;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.FindMissingBlobsRequest;
import build.bazel.remote.execution.v2.FindMissingBlobsResponse;
import com.google.devtools.build.lib.remote.SimpleBlobStoreActionCache;
import com.google.rpc.Code;
import io.grpc.stub.StreamObserver;
import java.io.IOException;

/** A basic implementation of a {@link ContentAddressableStorageImplBase} service. */
final class CasServer extends ContentAddressableStorageImplBase {
  static final long MAX_BATCH_SIZE_BYTES = 1024 * 1024 * 4;
  private final SimpleBlobStoreActionCache cache;

  public CasServer(SimpleBlobStoreActionCache cache) {
    this.cache = cache;
  }

  @Override
  public void findMissingBlobs(
      FindMissingBlobsRequest request, StreamObserver<FindMissingBlobsResponse> responseObserver) {
    FindMissingBlobsResponse.Builder response = FindMissingBlobsResponse.newBuilder();

    try {
      for (Digest digest : request.getBlobDigestsList()) {
        try {
          if (!cache.containsKey(digest)) {
            response.addMissingBlobDigests(digest);
          }
         } catch (InterruptedException e) {
          responseObserver.onError(StatusUtils.interruptedError(digest));
          Thread.currentThread().interrupt();
          return;
         }
      }
      responseObserver.onNext(response.build());
      responseObserver.onCompleted();
    } catch (IOException e) {
      responseObserver.onError(StatusUtils.internalError(e));
    }
  }

  @Override
  public void batchUpdateBlobs(
      BatchUpdateBlobsRequest request, StreamObserver<BatchUpdateBlobsResponse> responseObserver) {
    BatchUpdateBlobsResponse.Builder batchResponse = BatchUpdateBlobsResponse.newBuilder();
    for (BatchUpdateBlobsRequest.Request r : request.getRequestsList()) {
      BatchUpdateBlobsResponse.Response.Builder resp = batchResponse.addResponsesBuilder();
      try {
        Digest digest = cache.uploadBlob(r.getData().toByteArray());
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
}

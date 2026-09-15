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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;

import build.bazel.remote.execution.v2.Digest;
import com.google.bytestream.ByteStreamGrpc.ByteStreamImplBase;
import com.google.bytestream.ByteStreamProto.ReadRequest;
import com.google.bytestream.ByteStreamProto.ReadResponse;
import com.google.common.collect.ImmutableMap;
import com.google.protobuf.ByteString;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import java.util.HashMap;
import java.util.Map;

class FakeImmutableCacheByteStreamImpl extends ByteStreamImplBase {
  private final Map<ReadRequest, ReadResponse> cannedReplies;
  private final Map<ReadRequest, Integer> numErrors;
  // Start returning the correct response after this number of errors is reached.
  private static final int MAX_ERRORS = 3;

  public FakeImmutableCacheByteStreamImpl(Map<Digest, Object> contents) {
    ImmutableMap.Builder<ReadRequest, ReadResponse> b = ImmutableMap.builder();
    for (Map.Entry<Digest, Object> e : contents.entrySet()) {
      Object obj = e.getValue();
      ByteString data;
      if (obj instanceof String string) {
        data = ByteString.copyFromUtf8(string);
      } else if (obj instanceof ByteString byteString) {
        data = byteString;
      } else {
        throw new AssertionError(
            "expected object to be either a String or a ByteString, got a "
                + obj.getClass().getCanonicalName());
      }
      b.put(
          ReadRequest.newBuilder()
              .setResourceName("blobs/" + e.getKey().getHash() + "/" + e.getKey().getSizeBytes())
              .build(),
          ReadResponse.newBuilder().setData(data).build());
    }
    cannedReplies = b.build();
    numErrors = new HashMap<>();
  }

  public FakeImmutableCacheByteStreamImpl(Digest digest, String contents) {
    this(ImmutableMap.of(digest, contents));
  }

  public FakeImmutableCacheByteStreamImpl(Digest d1, String c1, Digest d2, String c2) {
    this(ImmutableMap.of(d1, c1, d2, c2));
  }

  @Override
  public void read(ReadRequest request, StreamObserver<ReadResponse> responseObserver) {
    assertThat(cannedReplies.keySet()).contains(request);
    int errCount = numErrors.getOrDefault(request, 0);
    if (errCount < MAX_ERRORS) {
      numErrors.put(request, errCount + 1);
      responseObserver.onError(Status.UNAVAILABLE.asRuntimeException());  // Retriable error.
    } else {
      responseObserver.onNext(cannedReplies.get(request));
      responseObserver.onCompleted();
    }
  }
}

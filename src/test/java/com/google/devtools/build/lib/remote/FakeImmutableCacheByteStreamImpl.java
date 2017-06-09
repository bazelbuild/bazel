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

import com.google.bytestream.ByteStreamGrpc.ByteStreamImplBase;
import com.google.bytestream.ByteStreamProto.ReadRequest;
import com.google.bytestream.ByteStreamProto.ReadResponse;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.protobuf.ByteString;
import io.grpc.stub.StreamObserver;
import java.util.Map;

class FakeImmutableCacheByteStreamImpl extends ByteStreamImplBase {
  private final Map<ReadRequest, ReadResponse> cannedReplies;

  public FakeImmutableCacheByteStreamImpl(Map<Digest, String> contents) {
    ImmutableMap.Builder<ReadRequest, ReadResponse> b = ImmutableMap.builder();
    for (Map.Entry<Digest, String> e : contents.entrySet()) {
      b.put(
          ReadRequest.newBuilder()
              .setResourceName("blobs/" + e.getKey().getHash() + "/" + e.getKey().getSizeBytes())
              .build(),
          ReadResponse.newBuilder().setData(ByteString.copyFromUtf8(e.getValue())).build());
    }
    cannedReplies = b.build();
  }

  public FakeImmutableCacheByteStreamImpl(Digest digest, String contents) {
    this(ImmutableMap.of(digest, contents));
  }

  public FakeImmutableCacheByteStreamImpl(Digest d1, String c1, Digest d2, String c2) {
    this(ImmutableMap.of(d1, c1, d2, c2));
  }

  @Override
  public void read(ReadRequest request, StreamObserver<ReadResponse> responseObserver) {
    assertThat(cannedReplies.containsKey(request)).isTrue();
    responseObserver.onNext(cannedReplies.get(request));
    responseObserver.onCompleted();
  }
}

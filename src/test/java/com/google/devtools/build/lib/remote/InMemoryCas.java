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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.remote.RemoteProtocol.BlobChunk;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadBlobRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadTreeMetadataReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadTreeMetadataRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasLookupReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasLookupRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasStatus;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadBlobReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadBlobRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadTreeMetadataReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadTreeMetadataRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ContentDigest;
import com.google.protobuf.ByteString;
import io.grpc.stub.StreamObserver;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/** An in-memory implementation of GrpcCasInterface. */
final class InMemoryCas implements GrpcCasInterface {
  private final Map<ByteString, ByteString> content = new HashMap<>();

  public ContentDigest put(byte[] data) {
    ContentDigest digest = ContentDigests.computeDigest(data);
    ByteString key = digest.getDigest();
    ByteString value = ByteString.copyFrom(data);
    content.put(key, value);
    return digest;
  }

  @Override
  public CasLookupReply lookup(CasLookupRequest request) {
    CasStatus.Builder result = CasStatus.newBuilder();
    for (ContentDigest digest : request.getDigestList()) {
      ByteString key = digest.getDigest();
      if (!content.containsKey(key)) {
        result.addMissingDigest(digest);
      }
    }
    if (result.getMissingDigestCount() != 0) {
      result.setError(CasStatus.ErrorCode.MISSING_DIGEST);
    } else {
      result.setSucceeded(true);
    }
    return CasLookupReply.newBuilder().setStatus(result).build();
  }

  @Override
  public CasUploadTreeMetadataReply uploadTreeMetadata(CasUploadTreeMetadataRequest request) {
    return CasUploadTreeMetadataReply.newBuilder()
        .setStatus(CasStatus.newBuilder().setSucceeded(true))
        .build();
  }

  @Override
  public CasDownloadTreeMetadataReply downloadTreeMetadata(
      CasDownloadTreeMetadataRequest request) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Iterator<CasDownloadReply> downloadBlob(CasDownloadBlobRequest request) {
    List<CasDownloadReply> result = new ArrayList<>();
    for (ContentDigest digest : request.getDigestList()) {
      CasDownloadReply.Builder builder = CasDownloadReply.newBuilder();
      ByteString item = content.get(digest.getDigest());
      if (item != null) {
        builder.setStatus(CasStatus.newBuilder().setSucceeded(true));
        builder.setData(BlobChunk.newBuilder().setData(item).setDigest(digest));
      } else {
        throw new IllegalStateException();
      }
      result.add(builder.build());
    }
    return result.iterator();
  }

  @Override
  public StreamObserver<CasUploadBlobRequest> uploadBlobAsync(
      final StreamObserver<CasUploadBlobReply> responseObserver) {
    return new StreamObserver<CasUploadBlobRequest>() {
      private ContentDigest digest;
      private ByteArrayOutputStream current;

      @Override
      public void onNext(CasUploadBlobRequest value) {
        BlobChunk chunk = value.getData();
        if (chunk.hasDigest()) {
          Preconditions.checkState(digest == null);
          digest = chunk.getDigest();
          current = new ByteArrayOutputStream();
        }
        try {
          current.write(chunk.getData().toByteArray());
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
        responseObserver.onNext(
            CasUploadBlobReply.newBuilder()
                .setStatus(CasStatus.newBuilder().setSucceeded(true))
                .build());
      }

      @Override
      public void onError(Throwable t) {
        throw new RuntimeException(t);
      }

      @Override
      public void onCompleted() {
        ContentDigest check = ContentDigests.computeDigest(current.toByteArray());
        Preconditions.checkState(check.equals(digest), "%s != %s", digest, check);
        ByteString key = digest.getDigest();
        ByteString value = ByteString.copyFrom(current.toByteArray());
        digest = null;
        current = null;
        content.put(key, value);
        responseObserver.onCompleted();
      }
    };
  }
}
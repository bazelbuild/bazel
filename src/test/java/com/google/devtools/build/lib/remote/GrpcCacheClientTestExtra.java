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

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import build.bazel.remote.execution.v2.Digest;
import com.github.luben.zstd.Zstd;
import com.google.bytestream.ByteStreamGrpc.ByteStreamImplBase;
import com.google.bytestream.ByteStreamProto.ReadRequest;
import com.google.bytestream.ByteStreamProto.ReadResponse;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.common.options.Options;
import com.google.protobuf.ByteString;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import java.io.IOException;
import java.util.Arrays;
import org.junit.Test;

/** Extra tests for {@link GrpcCacheClient} that are not tested internally. */
public class GrpcCacheClientTestExtra extends GrpcCacheClientTestBase {

  @Test
  public void compressedDownloadBlobIsRetriedWithProgress()
      throws IOException, InterruptedException {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.cacheCompression = true;
    final GrpcCacheClient client = newClient(options);
    final Digest digest = DIGEST_UTIL.computeAsUtf8("abcdefg");
    ByteString chunk1 = ByteString.copyFrom(Zstd.compress("abc".getBytes(UTF_8)));
    ByteString chunk2 = ByteString.copyFrom(Zstd.compress("def".getBytes(UTF_8)));
    ByteString chunk3 = ByteString.copyFrom(Zstd.compress("g".getBytes(UTF_8)));
    serviceRegistry.addService(
        new ByteStreamImplBase() {
          private boolean first = true;

          @Override
          public void read(ReadRequest request, StreamObserver<ReadResponse> responseObserver) {
            assertThat(request.getResourceName().contains(digest.getHash())).isTrue();
            if (first) {
              first = false;
              responseObserver.onError(Status.DEADLINE_EXCEEDED.asException());
              return;
            }
            switch (Math.toIntExact(request.getReadOffset())) {
              case 0:
                responseObserver.onNext(ReadResponse.newBuilder().setData(chunk1).build());
                break;
              case 3:
                responseObserver.onNext(ReadResponse.newBuilder().setData(chunk2).build());
                break;
              case 6:
                responseObserver.onNext(ReadResponse.newBuilder().setData(chunk3).build());
                responseObserver.onCompleted();
                return;
              default:
                throw new IllegalStateException("unexpected offset " + request.getReadOffset());
            }
            responseObserver.onError(Status.DEADLINE_EXCEEDED.asException());
          }
        });
    assertThat(new String(downloadBlob(context, client, digest), UTF_8)).isEqualTo("abcdefg");
  }

  @Test
  public void testCompressedDownload() throws IOException, InterruptedException {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.cacheCompression = true;
    final GrpcCacheClient client = newClient(options);
    final byte[] data = "abcdefg".getBytes(UTF_8);
    final Digest digest = DIGEST_UTIL.compute(data);
    final byte[] compressed = Zstd.compress(data);

    serviceRegistry.addService(
        new ByteStreamImplBase() {
          @Override
          public void read(ReadRequest request, StreamObserver<ReadResponse> responseObserver) {
            assertThat(request.getResourceName().contains(digest.getHash())).isTrue();
            responseObserver.onNext(
                ReadResponse.newBuilder()
                    .setData(
                        ByteString.copyFrom(
                            Arrays.copyOfRange(compressed, 0, compressed.length / 3)))
                    .build());
            responseObserver.onNext(
                ReadResponse.newBuilder()
                    .setData(
                        ByteString.copyFrom(
                            Arrays.copyOfRange(
                                compressed, compressed.length / 3, compressed.length / 3 * 2)))
                    .build());
            responseObserver.onNext(
                ReadResponse.newBuilder()
                    .setData(
                        ByteString.copyFrom(
                            Arrays.copyOfRange(
                                compressed, compressed.length / 3 * 2, compressed.length)))
                    .build());
            responseObserver.onCompleted();
          }
        });
    assertThat(downloadBlob(context, client, digest)).isEqualTo(data);
  }
}

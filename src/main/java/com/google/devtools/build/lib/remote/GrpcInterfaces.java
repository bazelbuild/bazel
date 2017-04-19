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

import com.google.devtools.build.lib.remote.CasServiceGrpc.CasServiceBlockingStub;
import com.google.devtools.build.lib.remote.CasServiceGrpc.CasServiceStub;
import com.google.devtools.build.lib.remote.ExecuteServiceGrpc.ExecuteServiceBlockingStub;
import com.google.devtools.build.lib.remote.ExecutionCacheServiceGrpc.ExecutionCacheServiceBlockingStub;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadBlobRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadTreeMetadataReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadTreeMetadataRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasLookupReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasLookupRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadBlobReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadBlobRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadTreeMetadataReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadTreeMetadataRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecuteReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecuteRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheSetReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheSetRequest;
import io.grpc.Channel;
import io.grpc.stub.StreamObserver;
import java.util.Iterator;
import java.util.concurrent.TimeUnit;

/** Implementations of the gRPC interfaces that actually talk to gRPC. */
public class GrpcInterfaces {
  /** Create a {@link GrpcCasInterface} instance that actually talks to gRPC. */
  public static GrpcCasInterface casInterface(
      final int grpcTimeoutSeconds,
      final Channel channel,
      final ChannelOptions channelOptions) {
    return new GrpcCasInterface() {
      private CasServiceBlockingStub getCasServiceBlockingStub() {
        return CasServiceGrpc.newBlockingStub(channel)
            .withCallCredentials(channelOptions.getCallCredentials())
            .withDeadlineAfter(grpcTimeoutSeconds, TimeUnit.SECONDS);
      }

      private CasServiceStub getCasServiceStub() {
        return CasServiceGrpc.newStub(channel)
            .withCallCredentials(channelOptions.getCallCredentials())
            .withDeadlineAfter(grpcTimeoutSeconds, TimeUnit.SECONDS);
      }

      @Override
      public CasLookupReply lookup(CasLookupRequest request) {
        return getCasServiceBlockingStub().lookup(request);
      }

      @Override
      public CasUploadTreeMetadataReply uploadTreeMetadata(CasUploadTreeMetadataRequest request) {
        return getCasServiceBlockingStub().uploadTreeMetadata(request);
      }

      @Override
      public CasDownloadTreeMetadataReply downloadTreeMetadata(
          CasDownloadTreeMetadataRequest request) {
        return getCasServiceBlockingStub().downloadTreeMetadata(request);
      }

      @Override
      public Iterator<CasDownloadReply> downloadBlob(CasDownloadBlobRequest request) {
        return getCasServiceBlockingStub().downloadBlob(request);
      }

      @Override
      public StreamObserver<CasUploadBlobRequest> uploadBlobAsync(
          StreamObserver<CasUploadBlobReply> responseObserver) {
        return getCasServiceStub().uploadBlob(responseObserver);
      }
    };
  }

  /** Create a {@link GrpcCasInterface} instance that actually talks to gRPC. */
  public static GrpcExecutionCacheInterface executionCacheInterface(
      final int grpcTimeoutSeconds,
      final Channel channel,
      final ChannelOptions channelOptions) {
    return new GrpcExecutionCacheInterface() {
      private ExecutionCacheServiceBlockingStub getExecutionCacheServiceBlockingStub() {
        return ExecutionCacheServiceGrpc.newBlockingStub(channel)
            .withCallCredentials(channelOptions.getCallCredentials())
            .withDeadlineAfter(grpcTimeoutSeconds, TimeUnit.SECONDS);
      }

      @Override
      public ExecutionCacheReply getCachedResult(ExecutionCacheRequest request) {
        return getExecutionCacheServiceBlockingStub().getCachedResult(request);
      }

      @Override
      public ExecutionCacheSetReply setCachedResult(ExecutionCacheSetRequest request) {
        return getExecutionCacheServiceBlockingStub().setCachedResult(request);
      }
    };
  }

  /** Create a {@link GrpcExecutionInterface} instance that actually talks to gRPC. */
  public static GrpcExecutionInterface executionInterface(
      final int grpcTimeoutSeconds,
      final Channel channel,
      final ChannelOptions channelOptions) {
    return new GrpcExecutionInterface() {
      @Override
      public Iterator<ExecuteReply> execute(ExecuteRequest request) {
        ExecuteServiceBlockingStub stub =
            ExecuteServiceGrpc.newBlockingStub(channel)
                .withCallCredentials(channelOptions.getCallCredentials())
                .withDeadlineAfter(
                    grpcTimeoutSeconds + request.getTimeoutMillis() / 1000, TimeUnit.SECONDS);
        return stub.execute(request);
      }
    };
  }
}

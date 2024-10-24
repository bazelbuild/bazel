// Copyright 2024 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.buildeventservice;

import static com.google.common.base.Preconditions.checkState;

import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceGrpcClient;
import com.google.devtools.build.lib.remote.util.FreePortFinder;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.v1.PublishBuildEventGrpc.PublishBuildEventImplBase;
import com.google.devtools.build.v1.PublishBuildToolEventStreamRequest;
import com.google.devtools.build.v1.PublishBuildToolEventStreamResponse;
import com.google.devtools.build.v1.PublishLifecycleEventRequest;
import com.google.protobuf.Empty;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import java.io.IOException;
import java.util.Collection;
import java.util.UUID;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for Bazel's {@link BuildEventServiceTransport} with a {@link BuildEventServiceGrpcClient}
 * transport.
 */
@RunWith(JUnit4.class)
public class BuildEventServiceTransportGrpcTest extends AbstractBuildEventServiceTransportTest {

  // This field is `public` to allow subclasses to override #createBesServer().
  public BuildEventRecorderGrpc server;

  @Override
  protected AbstractBuildEventRecorder createBesServer() {
    server = new BuildEventRecorderGrpc();
    return server;
  }

  @Override
  protected BuildEventServiceClient createBesClient() {
    checkState(server != null && server.getPort() > 0, "gRPC BES server not started.");
    return createBesClient(server.getPort());
  }

  @Override
  protected BuildEventServiceClient createBesClient(int serverPort) {
    return new BuildEventServiceGrpcClient(
        ManagedChannelBuilder.forTarget("localhost:" + serverPort).usePlaintext().build(),
        /* callCredentials= */ null,
        /* interceptor= */ null,
        "testing/" + UUID.randomUUID(),
        UUID.randomUUID());
  }

  @Override
  protected DigestHashFunction makeVfsHashFunction() {
    return DigestHashFunction.SHA256;
  }

  /**
   * A GRPC-protocol {@link AbstractBuildEventRecorder} that may be subclassed for alternative
   * testing scenarios.
   */
  public static class BuildEventRecorderGrpc extends AbstractBuildEventRecorder {

    protected Server server;

    private volatile boolean publishBuildToolEventStreamAccepted = false;

    @Override
    protected void startRpcServer(int port) {
      try {
        server =
            ServerBuilder.forPort(port)
                .addService(new BuildEventRecorderGrpc.BuildEventService())
                .build()
                .start();
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    @Override
    protected void stopRpcServer() {
      try {
        if (server != null) {
          server.shutdownNow();
          server.awaitTermination();
          server = null;
        }
      } catch (InterruptedException e) {
        throw new RuntimeException(e);
      }
    }

    @Override
    protected int getPort() {
      return server == null ? -1 : server.getPort();
    }

    @Override
    protected int pickNewPort() {
      try {
        return FreePortFinder.pickUnusedRandomPort();
      } catch (IOException | InterruptedException e) {
        throw new RuntimeException(e);
      }
    }

    /** Faked {@code PublishBuildEvent} service, for testing. */
    private class BuildEventService extends PublishBuildEventImplBase {

      @Override
      public void publishLifecycleEvent(
          PublishLifecycleEventRequest request, StreamObserver<Empty> streamObserver) {
        synchronized (BuildEventRecorderGrpc.this) {
          lifecycleEvents.put(request.getBuildEvent().getStreamId(), request);
          Status status = computeLifecycleResponse(request);
          if (status.isOk()) {
            streamObserver.onNext(Empty.getDefaultInstance());
            streamObserver.onCompleted();
          } else {
            streamObserver.onError(status.asException());
          }
        }
      }

      @Override
      public StreamObserver<PublishBuildToolEventStreamRequest> publishBuildToolEventStream(
          final StreamObserver<PublishBuildToolEventStreamResponse> stream) {
        publishBuildToolEventStreamAccepted = true;
        return new StreamObserver<PublishBuildToolEventStreamRequest>() {
          @Override
          public void onNext(PublishBuildToolEventStreamRequest request) {
            synchronized (BuildEventRecorderGrpc.this) {
              streamEvents.put(request.getOrderedBuildEvent().getStreamId(), request);
              if (sendOutOfOrderAcknowledgments) {
                stream.onNext(
                    PublishBuildToolEventStreamResponse.newBuilder()
                        .setStreamId(request.getOrderedBuildEvent().getStreamId())
                        .setSequenceNumber(request.getOrderedBuildEvent().getSequenceNumber() + 1)
                        .build());
                return;
              }
              Pair<Status, Collection<PublishBuildToolEventStreamResponse>> response =
                  computeStreamResponse(request);
              Status status = response.getFirst();
              if (status == null || status.isOk()) {
                successfulStreamEvents.put(request.getOrderedBuildEvent().getStreamId(), request);
                for (PublishBuildToolEventStreamResponse messages : response.getSecond()) {
                  stream.onNext(messages);
                }
                if (status != null && status.isOk()) {
                  stream.onCompleted();
                }
              } else {
                stream.onError(status.asException());
              }
            }
          }

          @Override
          public void onError(Throwable t) {
            eventStreamError = Status.fromThrowable(t);
            t.printStackTrace();
          }

          @Override
          public void onCompleted() {}
        };
      }
    }

    @Override
    protected boolean publishBuildToolEventStreamAccepted() {
      return publishBuildToolEventStreamAccepted;
    }
  }
}

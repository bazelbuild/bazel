// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceGrpcClient;
import com.google.devtools.build.v1.PublishBuildEventGrpc;
import com.google.devtools.build.v1.PublishBuildToolEventStreamRequest;
import com.google.devtools.build.v1.PublishBuildToolEventStreamResponse;
import io.grpc.ClientInterceptor;
import io.grpc.ManagedChannel;
import io.grpc.Metadata;
import io.grpc.Server;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.ServerInterceptors;
import io.grpc.ServerServiceDefinition;
import io.grpc.Status;
import io.grpc.StatusException;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.MetadataUtils;
import io.grpc.stub.StreamObserver;
import java.util.ArrayList;
import java.util.UUID;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link BuildEventServiceGrpcClient}. */
@RunWith(JUnit4.class)
public class BuildEventServiceGrpcClientTest {

  private static final PublishBuildEventGrpc.PublishBuildEventImplBase NOOP_SERVER =
      new PublishBuildEventGrpc.PublishBuildEventImplBase() {
        @Override
        public StreamObserver<PublishBuildToolEventStreamRequest> publishBuildToolEventStream(
            StreamObserver<PublishBuildToolEventStreamResponse> responseObserver) {
          responseObserver.onCompleted();
          return NULL_OBSERVER;
        }
      };

  private static final StreamObserver<PublishBuildToolEventStreamRequest> NULL_OBSERVER =
      new StreamObserver<PublishBuildToolEventStreamRequest>() {
        @Override
        public void onNext(PublishBuildToolEventStreamRequest value) {}

        @Override
        public void onError(Throwable t) {}

        @Override
        public void onCompleted() {}
      };

  private static final class TestServer implements AutoCloseable {
    private final Server server;
    private final ManagedChannel channel;

    TestServer(Server server, ManagedChannel channel) {
      this.server = server;
      this.channel = channel;
    }

    ManagedChannel getChannel() {
      return channel;
    }

    @Override
    public void close() {
      channel.shutdown();
      server.shutdown();
    }
  }

  /** Test helper that sets up a in-process test server. */
  private static TestServer startTestServer(ServerServiceDefinition service) throws Exception {
    String uniqueName = UUID.randomUUID().toString();
    Server server =
        InProcessServerBuilder.forName(uniqueName).directExecutor().addService(service).build();
    server.start();
    return new TestServer(
        server, InProcessChannelBuilder.forName(uniqueName).directExecutor().build());
  }

  @Test
  public void besHeaders() throws Exception {
    ArrayList<Metadata> seenHeaders = new ArrayList<>();
    try (TestServer server =
        startTestServer(
            ServerInterceptors.intercept(
                NOOP_SERVER,
                new ServerInterceptor() {
                  @Override
                  public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(
                      ServerCall<ReqT, RespT> call,
                      Metadata headers,
                      ServerCallHandler<ReqT, RespT> next) {
                    synchronized (seenHeaders) {
                      seenHeaders.add(headers);
                    }
                    return next.startCall(call, headers);
                  }
                }))) {
      Metadata extraHeaders = new Metadata();
      extraHeaders.put(Metadata.Key.of("metadata-foo", Metadata.ASCII_STRING_MARSHALLER), "bar");
      ClientInterceptor interceptor = MetadataUtils.newAttachHeadersInterceptor(extraHeaders);
      BuildEventServiceGrpcClient grpcClient =
          new BuildEventServiceGrpcClient(
              server.getChannel(),
              null,
              interceptor,
              "testing/" + UUID.randomUUID(),
              UUID.randomUUID());
      assertThat(grpcClient.openStream(ack -> {}).getStatus().get()).isEqualTo(Status.OK);
      assertThat(seenHeaders).hasSize(1);
      Metadata headers = seenHeaders.get(0);
      assertThat(headers.get(Metadata.Key.of("metadata-foo", Metadata.ASCII_STRING_MARSHALLER)))
          .isEqualTo("bar");
    }
  }

  @Test
  public void immediateSuccess() throws Exception {
    try (TestServer server = startTestServer(NOOP_SERVER.bindService())) {
      assertThat(
              new BuildEventServiceGrpcClient(
                      server.getChannel(),
                      null,
                      null,
                      "testing/" + UUID.randomUUID(),
                      UUID.randomUUID())
                  .openStream(ack -> {})
                  .getStatus()
                  .get())
          .isEqualTo(Status.OK);
    }
  }

  @Test
  public void immediateFailure() throws Exception {
    try (TestServer server =
        startTestServer(
            new PublishBuildEventGrpc.PublishBuildEventImplBase() {
              @Override
              public StreamObserver<PublishBuildToolEventStreamRequest> publishBuildToolEventStream(
                  StreamObserver<PublishBuildToolEventStreamResponse> responseObserver) {
                responseObserver.onError(new StatusException(Status.INTERNAL));
                return NULL_OBSERVER;
              }
            }.bindService())) {
      assertThat(
              new BuildEventServiceGrpcClient(
                      server.getChannel(),
                      null,
                      null,
                      "testing/" + UUID.randomUUID(),
                      UUID.randomUUID())
                  .openStream(ack -> {})
                  .getStatus()
                  .get())
          .isEqualTo(Status.INTERNAL);
    }
  }
}

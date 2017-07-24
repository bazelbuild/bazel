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

package com.google.devtools.build.lib.buildeventservice.client;

import static com.google.common.base.Strings.isNullOrEmpty;
import static com.google.devtools.build.lib.util.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.util.Preconditions.checkState;
import static java.lang.System.getenv;
import static java.nio.file.Files.newInputStream;
import static java.util.concurrent.TimeUnit.MILLISECONDS;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.common.base.Function;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.v1.PublishBuildEventGrpc;
import com.google.devtools.build.v1.PublishBuildEventGrpc.PublishBuildEventBlockingStub;
import com.google.devtools.build.v1.PublishBuildEventGrpc.PublishBuildEventStub;
import com.google.devtools.build.v1.PublishBuildToolEventStreamRequest;
import com.google.devtools.build.v1.PublishBuildToolEventStreamResponse;
import com.google.devtools.build.v1.PublishLifecycleEventRequest;
import io.grpc.CallCredentials;
import io.grpc.ManagedChannel;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.auth.MoreCallCredentials;
import io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.NegotiationType;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.stub.AbstractStub;
import io.grpc.stub.StreamObserver;
import io.netty.handler.ssl.SslContext;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.Nullable;
import javax.net.ssl.SSLException;
import org.joda.time.Duration;

/** Implementation of BuildEventServiceClient that uploads data using gRPC. */
public class BuildEventServiceGrpcClient implements BuildEventServiceClient {

  private static final Logger logger =
      Logger.getLogger(BuildEventServiceGrpcClient.class.getName());

  /** Max wait time for a single non-streaming RPC to finish */
  private static final Duration RPC_TIMEOUT = Duration.standardSeconds(15);
  /** See https://developers.google.com/identity/protocols/application-default-credentials * */
  private static final String DEFAULT_APP_CREDENTIALS_ENV_VAR = "GOOGLE_APPLICATION_CREDENTIALS";
  /** TODO(eduardocolaco): Scope documentation.* */
  private static final String CREDENTIALS_SCOPE =
      "https://www.googleapis.com/auth/cloud-build-service";

  private final PublishBuildEventStub besAsync;
  private final PublishBuildEventBlockingStub besBlocking;
  private final ManagedChannel channel;
  private final AtomicReference<StreamObserver<PublishBuildToolEventStreamRequest>> streamReference;

  public BuildEventServiceGrpcClient(String serverSpec, boolean tlsEnabled,
      @Nullable String tlsCertificateFile, @Nullable String tlsAuthorityOverride,
      @Nullable String credentialsFile, @Nullable String credentialsScope) {
    this(getChannel(serverSpec, tlsEnabled, tlsCertificateFile, tlsAuthorityOverride),
        getCallCredentials(credentialsFile, credentialsScope));
  }

  public BuildEventServiceGrpcClient(
      ManagedChannel channel,
      @Nullable CallCredentials callCredentials) {
    this.channel = channel;
    this.besAsync = withCallCredentials(
        PublishBuildEventGrpc.newStub(channel), callCredentials);
    this.besBlocking = withCallCredentials(
        PublishBuildEventGrpc.newBlockingStub(channel), callCredentials);
    this.streamReference = new AtomicReference<>(null);
  }

  private static <T extends AbstractStub<T>> T withCallCredentials(
      T stub, @Nullable CallCredentials callCredentials) {
    stub = callCredentials != null ? stub.withCallCredentials(callCredentials) : stub;
    return stub;
  }

  @Override
  public Status publish(PublishLifecycleEventRequest lifecycleEvent) throws Exception {
    besBlocking
        .withDeadlineAfter(RPC_TIMEOUT.getMillis(), MILLISECONDS)
        .publishLifecycleEvent(lifecycleEvent);
    return Status.OK;
  }

  @Override
  public ListenableFuture<Status> openStream(
      Function<PublishBuildToolEventStreamResponse, Void> ack)
      throws Exception {
    SettableFuture<Status> streamFinished = SettableFuture.create();
    checkState(
        streamReference.compareAndSet(null, createStream(ack, streamFinished)),
        "Starting a new stream without closing the previous one");
    return streamFinished;
  }

  private StreamObserver<PublishBuildToolEventStreamRequest> createStream(
      final Function<PublishBuildToolEventStreamResponse, Void> ack,
      final SettableFuture<Status> streamFinished) {
    return besAsync.publishBuildToolEventStream(
        new StreamObserver<PublishBuildToolEventStreamResponse>() {
          @Override
          public void onNext(PublishBuildToolEventStreamResponse response) {
            ack.apply(response);
          }

          @Override
          public void onError(Throwable t) {
            streamReference.set(null);
            streamFinished.setException(t);
          }

          @Override
          public void onCompleted() {
            streamReference.set(null);
            streamFinished.set(Status.OK);
          }
        });
  }

  @Override
  public void sendOverStream(PublishBuildToolEventStreamRequest buildEvent) throws Exception {
    checkNotNull(streamReference.get(), "Attempting to send over a closed or unopened stream")
        .onNext(buildEvent);
  }

  @Override
  public void closeStream() {
    StreamObserver<PublishBuildToolEventStreamRequest> stream;
    if ((stream = streamReference.getAndSet(null)) != null) {
      stream.onCompleted();
    }
  }

  @Override
  public void abortStream(Status status) {
    StreamObserver<PublishBuildToolEventStreamRequest> stream;
    if ((stream = streamReference.getAndSet(null)) != null) {
      stream.onError(status.asException());
    }
  }

  @Override
  public boolean isStreamActive() {
    return streamReference.get() != null;
  }

  @Override
  public void shutdown() throws InterruptedException {
    this.channel.shutdown();
  }

  @Override
  public String userReadableError(Throwable t) {
    if (t instanceof StatusRuntimeException) {
      Throwable rootCause = Throwables.getRootCause(t);
      String message = ((StatusRuntimeException) t).getStatus().getCode().name();
      message += ": " + rootCause.getMessage();
      return message;
    } else {
      return t.getMessage();
    }
  }

  /**
   * Returns call credentials read from the specified file (if non-empty) or from
   * env(GOOGLE_APPLICATION_CREDENTIALS) otherwise.
   */
  @Nullable
  private static CallCredentials getCallCredentials(@Nullable String credentialsFile,
      @Nullable String credentialsScope) {
    String effectiveScope = credentialsScope != null ? credentialsScope : CREDENTIALS_SCOPE;
    try {
      if (!isNullOrEmpty(credentialsFile)) {
        return MoreCallCredentials.from(
            GoogleCredentials.fromStream(newInputStream(Paths.get(credentialsFile)))
                .createScoped(ImmutableList.of(effectiveScope)));

      } else if (!isNullOrEmpty(getenv(DEFAULT_APP_CREDENTIALS_ENV_VAR))) {
        return MoreCallCredentials.from(
            GoogleCredentials.getApplicationDefault()
                .createScoped(ImmutableList.of(effectiveScope)));
      }
    } catch (IOException e) {
      logger.log(Level.WARNING, "Failed to read credentials", e);
    }
    return null;
  }

  /**
   * Returns a ManagedChannel to the specified server.
   */
  private static ManagedChannel getChannel(String serverSpec, boolean tlsEnabled,
      @Nullable String tlsCertificateFile, @Nullable String tlsAuthorityOverride) {
    //TODO(buchgr): Use ManagedChannelBuilder once bazel uses a newer gRPC version.
    NettyChannelBuilder builder = NettyChannelBuilder.forTarget(serverSpec);
    builder.negotiationType(tlsEnabled ? NegotiationType.TLS : NegotiationType.PLAINTEXT);
    if (tlsCertificateFile != null) {
      try {
        SslContext sslContext =
            GrpcSslContexts.forClient().trustManager(new File(tlsCertificateFile)).build();
        builder.sslContext(sslContext);
      } catch (SSLException e) {
        throw new RuntimeException(e);
      }
    }
    if (tlsAuthorityOverride != null) {
      builder.overrideAuthority(tlsAuthorityOverride);
    }
    return builder.build();
  }
}

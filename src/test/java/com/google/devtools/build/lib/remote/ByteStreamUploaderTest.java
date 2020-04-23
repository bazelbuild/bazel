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
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.fail;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.bytestream.ByteStreamGrpc;
import com.google.bytestream.ByteStreamGrpc.ByteStreamImplBase;
import com.google.bytestream.ByteStreamProto.QueryWriteStatusRequest;
import com.google.bytestream.ByteStreamProto.QueryWriteStatusResponse;
import com.google.bytestream.ByteStreamProto.WriteRequest;
import com.google.bytestream.ByteStreamProto.WriteResponse;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.hash.HashCode;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TestUtils;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.protobuf.ByteString;
import io.grpc.BindableService;
import io.grpc.Context;
import io.grpc.ManagedChannel;
import io.grpc.Metadata;
import io.grpc.Server;
import io.grpc.ServerCall;
import io.grpc.ServerCall.Listener;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.ServerInterceptors;
import io.grpc.ServerServiceDefinition;
import io.grpc.Status;
import io.grpc.Status.Code;
import io.grpc.StatusRuntimeException;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.MetadataUtils;
import io.grpc.stub.StreamObserver;
import io.grpc.util.MutableHandlerRegistry;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/**
 * Tests for {@link ByteStreamUploader}.
 */
@RunWith(JUnit4.class)
public class ByteStreamUploaderTest {

  private static final DigestUtil DIGEST_UTIL = new DigestUtil(DigestHashFunction.SHA256);

  private static final int CHUNK_SIZE = 10;
  private static final String INSTANCE_NAME = "foo";

  private final MutableHandlerRegistry serviceRegistry = new MutableHandlerRegistry();
  private ListeningScheduledExecutorService retryService;

  private Server server;
  private ManagedChannel channel;
  private Context withEmptyMetadata;
  private Context prevContext;

  @Mock private Retrier.Backoff mockBackoff;

  @Before
  public final void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);

    String serverName = "Server for " + this.getClass();
    server = InProcessServerBuilder.forName(serverName).fallbackHandlerRegistry(serviceRegistry)
        .build().start();
    channel = InProcessChannelBuilder.forName(serverName).build();
    withEmptyMetadata =
        TracingMetadataUtils.contextWithMetadata(
            "none", "none", DIGEST_UTIL.asActionKey(Digest.getDefaultInstance()));

    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));

    // Needs to be repeated in every test that uses the timeout setting, since the tests run
    // on different threads than the setUp.
    prevContext = withEmptyMetadata.attach();
  }

  @After
  public void tearDown() throws Exception {
    // Needs to be repeated in every test that uses the timeout setting, since the tests run
    // on different threads than the tearDown.
    withEmptyMetadata.detach(prevContext);

    retryService.shutdownNow();
    retryService.awaitTermination(
        com.google.devtools.build.lib.testutil.TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS);

    channel.shutdownNow();
    channel.awaitTermination(5, TimeUnit.SECONDS);
    server.shutdownNow();
    server.awaitTermination();
  }

  @Test
  public void singleBlobUploadShouldWork() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(() -> mockBackoff, (e) -> true, retryService);
    ByteStreamUploader uploader =
        new ByteStreamUploader(
            INSTANCE_NAME,
            new ReferenceCountedChannel(channel),
            null, /* timeout seconds */
            60,
            retrier);

    byte[] blob = new byte[CHUNK_SIZE * 2 + 1];
    new Random().nextBytes(blob);

    Chunker chunker = Chunker.builder().setInput(blob).setChunkSize(CHUNK_SIZE).build();
    HashCode hash = HashCode.fromString(DIGEST_UTIL.compute(blob).getHash());

    serviceRegistry.addService(new ByteStreamImplBase() {
          @Override
          public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> streamObserver) {
            return new StreamObserver<WriteRequest>() {

              byte[] receivedData = new byte[blob.length];
              long nextOffset = 0;

              @Override
              public void onNext(WriteRequest writeRequest) {
                if (nextOffset == 0) {
                  assertThat(writeRequest.getResourceName()).isNotEmpty();
                  assertThat(writeRequest.getResourceName()).startsWith(INSTANCE_NAME + "/uploads");
                  assertThat(writeRequest.getResourceName()).endsWith(String.valueOf(blob.length));
                } else {
                  assertThat(writeRequest.getResourceName()).isEmpty();
                }

                assertThat(writeRequest.getWriteOffset()).isEqualTo(nextOffset);

                ByteString data = writeRequest.getData();

                System.arraycopy(data.toByteArray(), 0, receivedData, (int) nextOffset,
                    data.size());

                nextOffset += data.size();
                boolean lastWrite = blob.length == nextOffset;
                assertThat(writeRequest.getFinishWrite()).isEqualTo(lastWrite);
              }

              @Override
              public void onError(Throwable throwable) {
                fail("onError should never be called.");
              }

              @Override
              public void onCompleted() {
                assertThat(nextOffset).isEqualTo(blob.length);
                assertThat(receivedData).isEqualTo(blob);

                WriteResponse response =
                    WriteResponse.newBuilder().setCommittedSize(nextOffset).build();
                streamObserver.onNext(response);
                streamObserver.onCompleted();
              }
            };
          }
        });

    uploader.uploadBlob(hash, chunker, true);

    // This test should not have triggered any retries.
    Mockito.verifyZeroInteractions(mockBackoff);

    blockUntilInternalStateConsistent(uploader);

    withEmptyMetadata.detach(prevContext);
  }

  @Test
  public void progressiveUploadShouldWork() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    Mockito.when(mockBackoff.getRetryAttempts()).thenReturn(0);
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(() -> mockBackoff, (e) -> true, retryService);
    ByteStreamUploader uploader =
        new ByteStreamUploader(
            INSTANCE_NAME, new ReferenceCountedChannel(channel), null, 3, retrier);

    byte[] blob = new byte[CHUNK_SIZE * 2 + 1];
    new Random().nextBytes(blob);

    Chunker chunker = Chunker.builder().setInput(blob).setChunkSize(CHUNK_SIZE).build();
    HashCode hash = HashCode.fromString(DIGEST_UTIL.compute(blob).getHash());

    serviceRegistry.addService(
        new ByteStreamImplBase() {

          byte[] receivedData = new byte[blob.length];
          String receivedResourceName = null;
          boolean receivedComplete = false;
          long nextOffset = 0;
          long initialOffset = 0;
          boolean mustQueryWriteStatus = false;

          @Override
          public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> streamObserver) {
            return new StreamObserver<WriteRequest>() {
              @Override
              public void onNext(WriteRequest writeRequest) {
                assertThat(mustQueryWriteStatus).isFalse();

                String resourceName = writeRequest.getResourceName();
                if (nextOffset == initialOffset) {
                  if (initialOffset == 0) {
                    receivedResourceName = resourceName;
                  }
                  assertThat(resourceName).startsWith(INSTANCE_NAME + "/uploads");
                  assertThat(resourceName).endsWith(String.valueOf(blob.length));
                } else {
                  assertThat(resourceName).isEmpty();
                }

                assertThat(writeRequest.getWriteOffset()).isEqualTo(nextOffset);

                ByteString data = writeRequest.getData();

                System.arraycopy(
                    data.toByteArray(), 0, receivedData, (int) nextOffset, data.size());

                nextOffset += data.size();
                receivedComplete = blob.length == nextOffset;
                assertThat(writeRequest.getFinishWrite()).isEqualTo(receivedComplete);

                if (initialOffset == 0) {
                  streamObserver.onError(Status.DEADLINE_EXCEEDED.asException());
                  mustQueryWriteStatus = true;
                  initialOffset = nextOffset;
                }
              }

              @Override
              public void onError(Throwable throwable) {
                fail("onError should never be called.");
              }

              @Override
              public void onCompleted() {
                assertThat(nextOffset).isEqualTo(blob.length);
                assertThat(receivedData).isEqualTo(blob);

                WriteResponse response =
                    WriteResponse.newBuilder().setCommittedSize(nextOffset).build();
                streamObserver.onNext(response);
                streamObserver.onCompleted();
              }
            };
          }

          @Override
          public void queryWriteStatus(
              QueryWriteStatusRequest request, StreamObserver<QueryWriteStatusResponse> response) {
            String resourceName = request.getResourceName();
            final long committedSize;
            final boolean complete;
            if (receivedResourceName != null && receivedResourceName.equals(resourceName)) {
              assertThat(mustQueryWriteStatus).isTrue();
              mustQueryWriteStatus = false;
              committedSize = nextOffset;
              complete = receivedComplete;
            } else {
              committedSize = 0;
              complete = false;
            }
            response.onNext(
                QueryWriteStatusResponse.newBuilder()
                    .setCommittedSize(committedSize)
                    .setComplete(complete)
                    .build());
            response.onCompleted();
          }
        });

    uploader.uploadBlob(hash, chunker, true);

    // This test should not have triggered any retries.
    Mockito.verify(mockBackoff, Mockito.never()).nextDelayMillis();
    Mockito.verify(mockBackoff, Mockito.times(1)).getRetryAttempts();

    blockUntilInternalStateConsistent(uploader);

    withEmptyMetadata.detach(prevContext);
  }

  @Test
  public void concurrentlyCompletedUploadIsNotRetried() throws Exception {
    // Test that after an upload has failed and the QueryWriteStatus call returns
    // that the upload has completed that we'll not retry the upload.
    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(() -> new FixedBackoff(1, 0), (e) -> true, retryService);
    ByteStreamUploader uploader =
        new ByteStreamUploader(
            INSTANCE_NAME, new ReferenceCountedChannel(channel), null, 1, retrier);

    byte[] blob = new byte[CHUNK_SIZE * 2 + 1];
    new Random().nextBytes(blob);

    Chunker chunker = Chunker.builder().setInput(blob).setChunkSize(CHUNK_SIZE).build();
    HashCode hash = HashCode.fromString(DIGEST_UTIL.compute(blob).getHash());

    AtomicInteger numWriteCalls = new AtomicInteger(0);

    serviceRegistry.addService(
        new ByteStreamImplBase() {
          @Override
          public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> streamObserver) {
            numWriteCalls.getAndIncrement();
            streamObserver.onError(Status.DEADLINE_EXCEEDED.asException());
            return new StreamObserver<WriteRequest>() {
              @Override
              public void onNext(WriteRequest writeRequest) {}

              @Override
              public void onError(Throwable throwable) {}

              @Override
              public void onCompleted() {}
            };
          }

          @Override
          public void queryWriteStatus(
              QueryWriteStatusRequest request, StreamObserver<QueryWriteStatusResponse> response) {
            response.onNext(
                QueryWriteStatusResponse.newBuilder()
                    .setCommittedSize(blob.length)
                    .setComplete(true)
                    .build());
            response.onCompleted();
          }
        });

    uploader.uploadBlob(hash, chunker, true);

    // This test should not have triggered any retries.
    assertThat(numWriteCalls.get()).isEqualTo(1);

    blockUntilInternalStateConsistent(uploader);

    withEmptyMetadata.detach(prevContext);
  }

  @Test
  public void unimplementedQueryShouldRestartUpload() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    Mockito.when(mockBackoff.getRetryAttempts()).thenReturn(0);
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(() -> mockBackoff, (e) -> true, retryService);
    ByteStreamUploader uploader =
        new ByteStreamUploader(
            INSTANCE_NAME, new ReferenceCountedChannel(channel), null, 3, retrier);

    byte[] blob = new byte[CHUNK_SIZE * 2 + 1];
    new Random().nextBytes(blob);

    Chunker chunker = Chunker.builder().setInput(blob).setChunkSize(CHUNK_SIZE).build();
    HashCode hash = HashCode.fromString(DIGEST_UTIL.compute(blob).getHash());

    serviceRegistry.addService(
        new ByteStreamImplBase() {
          boolean expireCall = true;
          boolean sawReset = false;

          @Override
          public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> streamObserver) {
            return new StreamObserver<WriteRequest>() {
              @Override
              public void onNext(WriteRequest writeRequest) {
                if (expireCall) {
                  streamObserver.onError(Status.DEADLINE_EXCEEDED.asException());
                  expireCall = false;
                } else if (!sawReset && writeRequest.getWriteOffset() != 0) {
                  streamObserver.onError(Status.INVALID_ARGUMENT.asException());
                } else {
                  sawReset = true;
                  if (writeRequest.getFinishWrite()) {
                    long committedSize =
                        writeRequest.getWriteOffset() + writeRequest.getData().size();
                    streamObserver.onNext(
                        WriteResponse.newBuilder().setCommittedSize(committedSize).build());
                    streamObserver.onCompleted();
                  }
                }
              }

              @Override
              public void onError(Throwable throwable) {
                fail("onError should never be called.");
              }

              @Override
              public void onCompleted() {}
            };
          }

          @Override
          public void queryWriteStatus(
              QueryWriteStatusRequest request, StreamObserver<QueryWriteStatusResponse> response) {
            response.onError(Status.UNIMPLEMENTED.asException());
          }
        });

    uploader.uploadBlob(hash, chunker, true);

    // This test should have triggered a single retry, because it made
    // no progress.
    Mockito.verify(mockBackoff, Mockito.times(1)).nextDelayMillis();

    blockUntilInternalStateConsistent(uploader);

    withEmptyMetadata.detach(prevContext);
  }

  @Test
  public void earlyWriteResponseShouldCompleteUpload() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(() -> mockBackoff, (e) -> true, retryService);
    ByteStreamUploader uploader =
        new ByteStreamUploader(
            INSTANCE_NAME, new ReferenceCountedChannel(channel), null, 3, retrier);

    byte[] blob = new byte[CHUNK_SIZE * 2 + 1];
    new Random().nextBytes(blob);
    // provide only enough data to write a single chunk
    InputStream in = new ByteArrayInputStream(blob, 0, CHUNK_SIZE);

    Chunker chunker = Chunker.builder().setInput(blob.length, in).setChunkSize(CHUNK_SIZE).build();
    HashCode hash = HashCode.fromString(DIGEST_UTIL.compute(blob).getHash());

    serviceRegistry.addService(
        new ByteStreamImplBase() {
          @Override
          public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> streamObserver) {
            streamObserver.onNext(WriteResponse.newBuilder().setCommittedSize(blob.length).build());
            streamObserver.onCompleted();
            return new NoopStreamObserver();
          }
        });

    uploader.uploadBlob(hash, chunker, true);

    // This test should not have triggered any retries.
    Mockito.verifyZeroInteractions(mockBackoff);

    blockUntilInternalStateConsistent(uploader);

    withEmptyMetadata.detach(prevContext);
  }

  @Test
  public void incorrectCommittedSizeFailsUpload() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(() -> mockBackoff, (e) -> true, retryService);
    ByteStreamUploader uploader =
        new ByteStreamUploader(
            INSTANCE_NAME, new ReferenceCountedChannel(channel), null, 3, retrier);

    byte[] blob = new byte[CHUNK_SIZE * 2 + 1];
    new Random().nextBytes(blob);

    Chunker chunker = Chunker.builder().setInput(blob).setChunkSize(CHUNK_SIZE).build();
    HashCode hash = HashCode.fromString(DIGEST_UTIL.compute(blob).getHash());

    serviceRegistry.addService(
        new ByteStreamImplBase() {
          @Override
          public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> streamObserver) {
            streamObserver.onNext(
                WriteResponse.newBuilder().setCommittedSize(blob.length + 1).build());
            streamObserver.onCompleted();
            return new NoopStreamObserver();
          }
        });

    try {
      uploader.uploadBlob(hash, chunker, true);
      fail("Should have thrown an exception.");
    } catch (IOException e) {
      // expected
    }

    // This test should not have triggered any retries.
    Mockito.verifyZeroInteractions(mockBackoff);

    blockUntilInternalStateConsistent(uploader);

    withEmptyMetadata.detach(prevContext);
  }

  @Test
  public void multipleBlobsUploadShouldWork() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(() -> new FixedBackoff(1, 0), (e) -> true, retryService);
    ByteStreamUploader uploader =
        new ByteStreamUploader(
            INSTANCE_NAME,
            new ReferenceCountedChannel(channel),
            null, /* timeout seconds */
            60,
            retrier);

    int numUploads = 10;
    Map<HashCode, byte[]> blobsByHash = Maps.newHashMap();
    Map<HashCode, Chunker> chunkers = Maps.newHashMapWithExpectedSize(numUploads);
    Random rand = new Random();
    for (int i = 0; i < numUploads; i++) {
      int blobSize = rand.nextInt(CHUNK_SIZE * 10) + CHUNK_SIZE;
      byte[] blob = new byte[blobSize];
      rand.nextBytes(blob);
      Chunker chunker = Chunker.builder().setInput(blob).setChunkSize(CHUNK_SIZE).build();
      HashCode hash = HashCode.fromString(DIGEST_UTIL.compute(blob).getHash());
      chunkers.put(hash, chunker);
      blobsByHash.put(hash, blob);
    }

    serviceRegistry.addService(new MaybeFailOnceUploadService(blobsByHash));

    uploader.uploadBlobs(chunkers, true);

    blockUntilInternalStateConsistent(uploader);

    withEmptyMetadata.detach(prevContext);
  }

  @Test
  public void contextShouldBePreservedUponRetries() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    // We upload blobs with different context, and retry 3 times for each upload.
    // We verify that the correct metadata is passed to the server with every blob.
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(() -> new FixedBackoff(5, 0), (e) -> true, retryService);
    ByteStreamUploader uploader =
        new ByteStreamUploader(
            INSTANCE_NAME,
            new ReferenceCountedChannel(channel),
            null, /* timeout seconds */
            60,
            retrier);

    List<String> toUpload = ImmutableList.of("aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc");
    Map<Digest, Chunker> chunkers = Maps.newHashMapWithExpectedSize(toUpload.size());
    Map<String, Integer> uploadsFailed = Maps.newHashMap();
    for (String s : toUpload) {
      Chunker chunker = Chunker.builder().setInput(s.getBytes(UTF_8)).setChunkSize(3).build();
      Digest digest = DIGEST_UTIL.computeAsUtf8(s);
      chunkers.put(digest, chunker);
      uploadsFailed.put(digest.getHash(), 0);
    }

    BindableService bsService =
        new ByteStreamImplBase() {
          @Override
          public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> response) {
            return new StreamObserver<WriteRequest>() {

              private String digestHash;

              @Override
              public void onNext(WriteRequest writeRequest) {
                String resourceName = writeRequest.getResourceName();
                if (!resourceName.isEmpty()) {
                  String[] components = resourceName.split("/");
                  assertThat(components).hasLength(6);
                  digestHash = components[4];
                }
                assertThat(digestHash).isNotNull();
                RequestMetadata meta = TracingMetadataUtils.fromCurrentContext();
                assertThat(meta.getCorrelatedInvocationsId()).isEqualTo("build-req-id");
                assertThat(meta.getToolInvocationId()).isEqualTo("command-id");
                assertThat(meta.getActionId()).isEqualTo(digestHash);
                assertThat(meta.getToolDetails().getToolName()).isEqualTo("bazel");
                assertThat(meta.getToolDetails().getToolVersion())
                    .isEqualTo(BlazeVersionInfo.instance().getVersion());
                synchronized (this) {
                  Integer numFailures = uploadsFailed.get(digestHash);
                  if (numFailures < 3) {
                    uploadsFailed.put(digestHash, numFailures + 1);
                    response.onError(Status.INTERNAL.asException());
                    return;
                  }
                }
              }

              @Override
              public void onError(Throwable throwable) {
                fail("onError should never be called.");
              }

              @Override
              public void onCompleted() {
                response.onNext(WriteResponse.newBuilder().setCommittedSize(10).build());
                response.onCompleted();
              }
            };
          }

          @Override
          public void queryWriteStatus(
              QueryWriteStatusRequest request, StreamObserver<QueryWriteStatusResponse> response) {
            response.onNext(
                QueryWriteStatusResponse.newBuilder()
                    .setCommittedSize(0)
                    .setComplete(false)
                    .build());
            response.onCompleted();
          }
        };
    serviceRegistry.addService(
        ServerInterceptors.intercept(
            bsService, new TracingMetadataUtils.ServerHeadersInterceptor()));

    List<ListenableFuture<Void>> uploads = new ArrayList<>();

    for (Map.Entry<Digest, Chunker> chunkerEntry : chunkers.entrySet()) {
      Digest actionDigest = chunkerEntry.getKey();
      Context ctx =
          TracingMetadataUtils.contextWithMetadata(
              "build-req-id", "command-id", DIGEST_UTIL.asActionKey(actionDigest));
      ctx.run(
          () ->
              uploads.add(
                  uploader.uploadBlobAsync(
                      HashCode.fromString(actionDigest.getHash()),
                      chunkerEntry.getValue(),
                      /* forceUpload=*/ true)));
    }

    for (ListenableFuture<Void> upload : uploads) {
      upload.get();
    }

    blockUntilInternalStateConsistent(uploader);

    withEmptyMetadata.detach(prevContext);
  }

  @Test
  public void customHeadersAreAttachedToRequest() throws Exception {
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(() -> new FixedBackoff(1, 0), (e) -> true, retryService);

    Metadata metadata = new Metadata();
    metadata.put(Metadata.Key.of("Key1", Metadata.ASCII_STRING_MARSHALLER), "Value1");
    metadata.put(Metadata.Key.of("Key2", Metadata.ASCII_STRING_MARSHALLER), "Value2");

    ByteStreamUploader uploader =
        new ByteStreamUploader(
            INSTANCE_NAME,
            new ReferenceCountedChannel(
                InProcessChannelBuilder.forName("Server for " + this.getClass())
                    .intercept(MetadataUtils.newAttachHeadersInterceptor(metadata))
                    .build()),
            null, /* timeout seconds */
            60,
            retrier);

    byte[] blob = new byte[CHUNK_SIZE];
    Chunker chunker = Chunker.builder().setInput(blob).setChunkSize(CHUNK_SIZE).build();
    HashCode hash = HashCode.fromString(DIGEST_UTIL.compute(blob).getHash());

    serviceRegistry.addService(
        ServerInterceptors.intercept(
            new ByteStreamImplBase() {
              @Override
              public StreamObserver<WriteRequest> write(
                  StreamObserver<WriteResponse> streamObserver) {
                return new StreamObserver<WriteRequest>() {
                  @Override
                  public void onNext(WriteRequest writeRequest) {}

                  @Override
                  public void onError(Throwable throwable) {
                    fail("onError should never be called.");
                  }

                  @Override
                  public void onCompleted() {
                    WriteResponse response =
                        WriteResponse.newBuilder().setCommittedSize(blob.length).build();
                    streamObserver.onNext(response);
                    streamObserver.onCompleted();
                  }
                };
              }
            },
            new ServerInterceptor() {
              @Override
              public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(
                  ServerCall<ReqT, RespT> call,
                  Metadata metadata,
                  ServerCallHandler<ReqT, RespT> next) {
                assertThat(metadata.get(Metadata.Key.of("Key1", Metadata.ASCII_STRING_MARSHALLER)))
                    .isEqualTo("Value1");
                assertThat(metadata.get(Metadata.Key.of("Key2", Metadata.ASCII_STRING_MARSHALLER)))
                    .isEqualTo("Value2");
                assertThat(metadata.get(Metadata.Key.of("Key3", Metadata.ASCII_STRING_MARSHALLER)))
                    .isEqualTo(null);
                return next.startCall(call, metadata);
              }
            }));

    uploader.uploadBlob(hash, chunker, true);
  }

  @Test
  public void sameBlobShouldNotBeUploadedTwice() throws Exception {
    // Test that uploading the same file concurrently triggers only one file upload.

    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(() -> mockBackoff, (e) -> true, retryService);
    ByteStreamUploader uploader =
        new ByteStreamUploader(
            INSTANCE_NAME,
            new ReferenceCountedChannel(channel),
            null, /* timeout seconds */
            60,
            retrier);

    byte[] blob = new byte[CHUNK_SIZE * 10];
    Chunker chunker = Chunker.builder().setInput(blob).setChunkSize(CHUNK_SIZE).build();
    HashCode hash = HashCode.fromString(DIGEST_UTIL.compute(blob).getHash());

    AtomicInteger numWriteCalls = new AtomicInteger();
    CountDownLatch blocker = new CountDownLatch(1);

    serviceRegistry.addService(new ByteStreamImplBase() {
      @Override
      public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> response) {
        numWriteCalls.incrementAndGet();
        try {
          // Ensures that the first upload does not finish, before the second upload is started.
          blocker.await();
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
        }

        return new StreamObserver<WriteRequest>() {

          private long bytesReceived;

          @Override
          public void onNext(WriteRequest writeRequest) {
            bytesReceived += writeRequest.getData().size();
          }

          @Override
          public void onError(Throwable throwable) {
            fail("onError should never be called.");
          }

          @Override
          public void onCompleted() {
            response.onNext(WriteResponse.newBuilder().setCommittedSize(bytesReceived).build());
            response.onCompleted();
          }
        };
      }
    });

    Future<?> upload1 = uploader.uploadBlobAsync(hash, chunker, true);
    Future<?> upload2 = uploader.uploadBlobAsync(hash, chunker, true);

    blocker.countDown();

    assertThat(upload1).isSameInstanceAs(upload2);

    upload1.get();

    assertThat(numWriteCalls.get()).isEqualTo(1);

    withEmptyMetadata.detach(prevContext);
  }

  @Test
  public void errorsShouldBeReported() throws IOException, InterruptedException {
    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(() -> new FixedBackoff(1, 10), (e) -> true, retryService);
    ByteStreamUploader uploader =
        new ByteStreamUploader(
            INSTANCE_NAME,
            new ReferenceCountedChannel(channel),
            null, /* timeout seconds */
            60,
            retrier);

    byte[] blob = new byte[CHUNK_SIZE];
    Chunker chunker = Chunker.builder().setInput(blob).setChunkSize(CHUNK_SIZE).build();
    HashCode hash = HashCode.fromString(DIGEST_UTIL.compute(blob).getHash());

    serviceRegistry.addService(new ByteStreamImplBase() {
      @Override
      public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> response) {
        response.onError(Status.INTERNAL.asException());
        return new NoopStreamObserver();
      }
    });

    try {
      uploader.uploadBlob(hash, chunker, true);
      fail("Should have thrown an exception.");
    } catch (IOException e) {
      assertThat(RemoteRetrierUtils.causedByStatus(e, Code.INTERNAL)).isTrue();
    }

    withEmptyMetadata.detach(prevContext);
  }

  @Test
  public void shutdownShouldCancelOngoingUploads() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(() -> new FixedBackoff(1, 10), (e) -> true, retryService);
    ByteStreamUploader uploader =
        new ByteStreamUploader(
            INSTANCE_NAME,
            new ReferenceCountedChannel(channel),
            null, /* timeout seconds */
            60,
            retrier);

    CountDownLatch cancellations = new CountDownLatch(2);

    ServerServiceDefinition service =
        ServerServiceDefinition.builder(ByteStreamGrpc.SERVICE_NAME)
            .addMethod(
                ByteStreamGrpc.getWriteMethod(),
                new ServerCallHandler<WriteRequest, WriteResponse>() {
                  @Override
                  public Listener<WriteRequest> startCall(
                      ServerCall<WriteRequest, WriteResponse> call, Metadata headers) {
                    // Don't request() any messages from the client, so that the client will be
                    // blocked
                    // on flow control and thus the call will sit there idle long enough to receive
                    // the
                    // cancellation.
                    return new Listener<WriteRequest>() {
                      @Override
                      public void onCancel() {
                        cancellations.countDown();
                      }
                    };
                  }
                })
            .build();

    serviceRegistry.addService(service);

    byte[] blob1 = new byte[CHUNK_SIZE];
    Chunker chunker1 = Chunker.builder().setInput(blob1).setChunkSize(CHUNK_SIZE).build();
    HashCode hash1 = HashCode.fromString(DIGEST_UTIL.compute(blob1).getHash());

    byte[] blob2 = new byte[CHUNK_SIZE + 1];
    Chunker chunker2 = Chunker.builder().setInput(blob2).setChunkSize(CHUNK_SIZE).build();
    HashCode hash2 = HashCode.fromString(DIGEST_UTIL.compute(blob2).getHash());

    ListenableFuture<Void> f1 = uploader.uploadBlobAsync(hash1, chunker1, true);
    ListenableFuture<Void> f2 = uploader.uploadBlobAsync(hash2, chunker2, true);

    assertThat(uploader.uploadsInProgress()).isTrue();

    uploader.shutdown();

    cancellations.await();

    assertThat(f1.isCancelled()).isTrue();
    assertThat(f2.isCancelled()).isTrue();

    blockUntilInternalStateConsistent(uploader);

    withEmptyMetadata.detach(prevContext);
  }

  @Test
  public void failureInRetryExecutorShouldBeHandled() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    ListeningScheduledExecutorService retryService =
        MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(() -> new FixedBackoff(1, 10), (e) -> true, retryService);
    ByteStreamUploader uploader =
        new ByteStreamUploader(
            INSTANCE_NAME,
            new ReferenceCountedChannel(channel),
            null, /* timeout seconds */
            60,
            retrier);

    serviceRegistry.addService(new ByteStreamImplBase() {
      @Override
      public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> response) {
        // Immediately fail the call, so that it is retried.
        response.onError(Status.ABORTED.asException());
        return new NoopStreamObserver();
      }
    });

    retryService.shutdownNow();
    // Random very high timeout, as the test will timeout by itself.
    retryService.awaitTermination(1, TimeUnit.DAYS);
    assertThat(retryService.isShutdown()).isTrue();

    byte[] blob = new byte[1];
    Chunker chunker = Chunker.builder().setInput(blob).setChunkSize(CHUNK_SIZE).build();
    HashCode hash = HashCode.fromString(DIGEST_UTIL.compute(blob).getHash());
    try {
      uploader.uploadBlob(hash, chunker, true);
      fail("Should have thrown an exception.");
    } catch (IOException e) {
      assertThat(e).hasCauseThat().isInstanceOf(RejectedExecutionException.class);
    }

    withEmptyMetadata.detach(prevContext);
  }

  @Test
  public void resourceNameWithoutInstanceName() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(() -> mockBackoff, (e) -> true, retryService);
    ByteStreamUploader uploader =
        new ByteStreamUploader(
            /* instanceName */ null,
            new ReferenceCountedChannel(channel),
            null, /* timeout seconds */
            60,
            retrier);

    serviceRegistry.addService(new ByteStreamImplBase() {
      @Override
      public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> response) {
        return new StreamObserver<WriteRequest>() {
          @Override
          public void onNext(WriteRequest writeRequest) {
            // Test that the resource name doesn't start with an instance name.
            assertThat(writeRequest.getResourceName()).startsWith("uploads/");
          }

          @Override
          public void onError(Throwable throwable) {

          }

          @Override
          public void onCompleted() {
            response.onNext(WriteResponse.newBuilder().setCommittedSize(1).build());
            response.onCompleted();
          }
        };
      }
    });

    byte[] blob = new byte[1];
    Chunker chunker = Chunker.builder().setInput(blob).setChunkSize(CHUNK_SIZE).build();
    HashCode hash = HashCode.fromString(DIGEST_UTIL.compute(blob).getHash());

    uploader.uploadBlob(hash, chunker, true);

    withEmptyMetadata.detach(prevContext);
  }

  @Test
  public void nonRetryableStatusShouldNotBeRetried() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), /* No Status is retriable. */ (e) -> false, retryService);
    ByteStreamUploader uploader =
        new ByteStreamUploader(
            /* instanceName */ null,
            new ReferenceCountedChannel(channel),
            null, /* timeout seconds */
            60,
            retrier);

    AtomicInteger numCalls = new AtomicInteger();

    serviceRegistry.addService(new ByteStreamImplBase() {
      @Override
      public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> response) {
        numCalls.incrementAndGet();
        response.onError(Status.INTERNAL.asException());
        return new NoopStreamObserver();
      }
    });

    byte[] blob = new byte[1];
    Chunker chunker = Chunker.builder().setInput(blob).setChunkSize(CHUNK_SIZE).build();
    HashCode hash = HashCode.fromString(DIGEST_UTIL.compute(blob).getHash());

    try {
      uploader.uploadBlob(hash, chunker, true);
      fail("Should have thrown an exception.");
    } catch (IOException e) {
      assertThat(numCalls.get()).isEqualTo(1);
    }

    withEmptyMetadata.detach(prevContext);
  }

  @Test
  public void failedUploadsShouldNotDeduplicate() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(() -> Retrier.RETRIES_DISABLED, (e) -> false, retryService);
    ByteStreamUploader uploader =
        new ByteStreamUploader(
            INSTANCE_NAME,
            new ReferenceCountedChannel(channel),
            null, /* timeout seconds */
            60,
            retrier);

    byte[] blob = new byte[CHUNK_SIZE * 2 + 1];
    new Random().nextBytes(blob);

    Chunker chunker = Chunker.builder().setInput(blob).setChunkSize(CHUNK_SIZE).build();
    HashCode hash = HashCode.fromString(DIGEST_UTIL.compute(blob).getHash());

    AtomicInteger numUploads = new AtomicInteger();
    serviceRegistry.addService(
        new ByteStreamImplBase() {
          boolean failRequest = true;

          @Override
          public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> streamObserver) {
            numUploads.incrementAndGet();
            return new StreamObserver<WriteRequest>() {
              long nextOffset = 0;

              @Override
              public void onNext(WriteRequest writeRequest) {
                if (failRequest) {
                  streamObserver.onError(Status.UNKNOWN.asException());
                  failRequest = false;
                } else {
                  nextOffset += writeRequest.getData().size();
                  boolean lastWrite = blob.length == nextOffset;
                  assertThat(writeRequest.getFinishWrite()).isEqualTo(lastWrite);
                }
              }

              @Override
              public void onError(Throwable throwable) {
                fail("onError should never be called.");
              }

              @Override
              public void onCompleted() {
                assertThat(nextOffset).isEqualTo(blob.length);

                WriteResponse response =
                    WriteResponse.newBuilder().setCommittedSize(nextOffset).build();
                streamObserver.onNext(response);
                streamObserver.onCompleted();
              }
            };
          }
        });

    StatusRuntimeException expected = null;
    try {
      // This should fail
      uploader.uploadBlob(hash, chunker, true);
    } catch (IOException e) {
      if (e.getCause() instanceof StatusRuntimeException) {
        expected = (StatusRuntimeException) e.getCause();
      }
    }
    assertThat(expected).isNotNull();
    assertThat(Status.fromThrowable(expected).getCode()).isEqualTo(Code.UNKNOWN);
    // This should trigger an upload.
    uploader.uploadBlob(hash, chunker, false);

    assertThat(numUploads.get()).isEqualTo(2);

    blockUntilInternalStateConsistent(uploader);

    withEmptyMetadata.detach(prevContext);
  }

  @Test
  public void deduplicationOfUploadsShouldWork() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(() -> mockBackoff, (e) -> true, retryService);
    ByteStreamUploader uploader =
        new ByteStreamUploader(
            INSTANCE_NAME,
            new ReferenceCountedChannel(channel),
            null, /* timeout seconds */
            60,
            retrier);

    byte[] blob = new byte[CHUNK_SIZE * 2 + 1];
    new Random().nextBytes(blob);

    Chunker chunker = Chunker.builder().setInput(blob).setChunkSize(CHUNK_SIZE).build();
    HashCode hash = HashCode.fromString(DIGEST_UTIL.compute(blob).getHash());

    AtomicInteger numUploads = new AtomicInteger();
    serviceRegistry.addService(new ByteStreamImplBase() {
      @Override
      public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> streamObserver) {
        numUploads.incrementAndGet();
        return new StreamObserver<WriteRequest>() {

          long nextOffset = 0;

          @Override
          public void onNext(WriteRequest writeRequest) {
            nextOffset += writeRequest.getData().size();
            boolean lastWrite = blob.length == nextOffset;
            assertThat(writeRequest.getFinishWrite()).isEqualTo(lastWrite);
          }

          @Override
          public void onError(Throwable throwable) {
            fail("onError should never be called.");
          }

          @Override
          public void onCompleted() {
            assertThat(nextOffset).isEqualTo(blob.length);

            WriteResponse response =
                WriteResponse.newBuilder().setCommittedSize(nextOffset).build();
            streamObserver.onNext(response);
            streamObserver.onCompleted();
          }
        };
      }
    });

    uploader.uploadBlob(hash, chunker, true);
    // This should not trigger an upload.
    uploader.uploadBlob(hash, chunker, false);

    assertThat(numUploads.get()).isEqualTo(1);

    // This test should not have triggered any retries.
    Mockito.verifyZeroInteractions(mockBackoff);

    blockUntilInternalStateConsistent(uploader);

    withEmptyMetadata.detach(prevContext);
  }

  private static class NoopStreamObserver implements StreamObserver<WriteRequest> {
    @Override
    public void onNext(WriteRequest writeRequest) {
    }

    @Override
    public void onError(Throwable throwable) {
    }

    @Override
    public void onCompleted() {
    }
  }

  static class FixedBackoff implements Retrier.Backoff {

    private final int maxRetries;
    private final int delayMillis;

    private int retries;

    public FixedBackoff(int maxRetries, int delayMillis) {
      this.maxRetries = maxRetries;
      this.delayMillis = delayMillis;
    }

    @Override
    public long nextDelayMillis() {
      if (retries < maxRetries) {
        retries++;
        return delayMillis;
      }
      return -1;
    }

    @Override
    public int getRetryAttempts() {
      return retries;
    }
  }

  /**
   * An byte stream service where an upload for a given blob may or may not fail on the first
   * attempt but is guaranteed to succeed on the second try.
   */
  static class MaybeFailOnceUploadService extends ByteStreamImplBase {

    private final Map<HashCode, byte[]> blobsByHash;
    private final Set<HashCode> uploadsFailedOnce = Collections.synchronizedSet(Sets.newHashSet());
    private final Random rand = new Random();

    MaybeFailOnceUploadService(Map<HashCode, byte[]> blobsByHash) {
      this.blobsByHash = blobsByHash;
    }

    @Override
    public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> response) {
      return new StreamObserver<WriteRequest>() {

        private HashCode digestHash;
        private byte[] receivedData;
        private long nextOffset;

        @Override
        public void onNext(WriteRequest writeRequest) {
          if (nextOffset == 0) {
            String resourceName = writeRequest.getResourceName();
            assertThat(resourceName).isNotEmpty();

            String[] components = resourceName.split("/");
            assertThat(components).hasLength(6);
            digestHash = HashCode.fromString(components[4]);
            assertThat(blobsByHash).containsKey(digestHash);
            receivedData = new byte[Integer.parseInt(components[5])];
          }
          assertThat(digestHash).isNotNull();
          // An upload for a given blob has a 10% chance to fail once during its lifetime.
          // This is to exercise the retry mechanism a bit.
          boolean shouldFail = rand.nextInt(10) == 0 && !uploadsFailedOnce.contains(digestHash);
          if (shouldFail) {
            uploadsFailedOnce.add(digestHash);
            response.onError(Status.INTERNAL.asException());
            return;
          }

          ByteString data = writeRequest.getData();
          System.arraycopy(data.toByteArray(), 0, receivedData, (int) nextOffset, data.size());
          nextOffset += data.size();

          boolean lastWrite = nextOffset == receivedData.length;
          assertThat(writeRequest.getFinishWrite()).isEqualTo(lastWrite);
        }

        @Override
        public void onError(Throwable throwable) {
          fail("onError should never be called.");
        }

        @Override
        public void onCompleted() {
          byte[] expectedBlob = blobsByHash.get(digestHash);
          assertThat(receivedData).isEqualTo(expectedBlob);

          WriteResponse writeResponse =
              WriteResponse.newBuilder().setCommittedSize(receivedData.length).build();

          response.onNext(writeResponse);
          response.onCompleted();
        }
      };
    }

    @Override
    public void queryWriteStatus(
        QueryWriteStatusRequest request, StreamObserver<QueryWriteStatusResponse> response) {
      // force the client to reset the write
      response.onNext(
          QueryWriteStatusResponse.newBuilder().setCommittedSize(0).setComplete(false).build());
      response.onCompleted();
    }
  }

  private void blockUntilInternalStateConsistent(ByteStreamUploader uploader) throws Exception {
    // Poll until all upload futures have been removed from the internal hash map. The polling is
    // necessary, as listeners are executed after Future.get() calls are notified about completion.
    while (uploader.uploadsInProgress()) {
      Thread.sleep(1);
    }
  }
}

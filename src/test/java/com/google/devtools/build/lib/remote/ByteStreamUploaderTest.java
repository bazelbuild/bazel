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

import com.google.bytestream.ByteStreamGrpc;
import com.google.bytestream.ByteStreamGrpc.ByteStreamImplBase;
import com.google.bytestream.ByteStreamProto.WriteRequest;
import com.google.bytestream.ByteStreamProto.WriteResponse;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.remote.Retrier.RetryException;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.vfs.FileSystem.HashFunction;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.devtools.remoteexecution.v1test.RequestMetadata;
import com.google.protobuf.ByteString;
import io.grpc.BindableService;
import io.grpc.Channel;
import io.grpc.Context;
import io.grpc.Metadata;
import io.grpc.Server;
import io.grpc.ServerCall;
import io.grpc.ServerCall.Listener;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptors;
import io.grpc.ServerServiceDefinition;
import io.grpc.Status;
import io.grpc.Status.Code;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import io.grpc.util.MutableHandlerRegistry;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
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

  private static final DigestUtil DIGEST_UTIL = new DigestUtil(HashFunction.SHA256);

  private static final int CHUNK_SIZE = 10;
  private static final String INSTANCE_NAME = "foo";

  private final MutableHandlerRegistry serviceRegistry = new MutableHandlerRegistry();
  private final ListeningScheduledExecutorService retryService =
      MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));

  private Server server;
  private Channel channel;
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
    // Needs to be repeated in every test that uses the timeout setting, since the tests run
    // on different threads than the setUp.
    prevContext = withEmptyMetadata.attach();
  }

  @After
  public void tearDown() throws Exception {
    // Needs to be repeated in every test that uses the timeout setting, since the tests run
    // on different threads than the tearDown.
    withEmptyMetadata.detach(prevContext);

    server.shutdownNow();
    retryService.shutdownNow();
    server.awaitTermination();
  }

  @Test(timeout = 10000)
  public void singleBlobUploadShouldWork() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        new RemoteRetrier(() -> mockBackoff, (e) -> true, Retrier.ALLOW_ALL_CALLS);
    ByteStreamUploader uploader =
        new ByteStreamUploader(INSTANCE_NAME, channel, null, 3, retrier, retryService);

    byte[] blob = new byte[CHUNK_SIZE * 2 + 1];
    new Random().nextBytes(blob);

    Chunker chunker = new Chunker(blob, CHUNK_SIZE, DIGEST_UTIL);

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

    uploader.uploadBlob(chunker);

    // This test should not have triggered any retries.
    Mockito.verifyZeroInteractions(mockBackoff);

    blockUntilInternalStateConsistent(uploader);

    withEmptyMetadata.detach(prevContext);
  }

  @Test(timeout = 20000)
  public void multipleBlobsUploadShouldWork() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        new RemoteRetrier(() -> new FixedBackoff(1, 0), (e) -> true, Retrier.ALLOW_ALL_CALLS);
    ByteStreamUploader uploader =
        new ByteStreamUploader(INSTANCE_NAME, channel, null, 3, retrier, retryService);

    int numUploads = 100;
    Map<String, byte[]> blobsByHash = new HashMap<>();
    List<Chunker> builders = new ArrayList<>(numUploads);
    Random rand = new Random();
    for (int i = 0; i < numUploads; i++) {
      int blobSize = rand.nextInt(CHUNK_SIZE * 10) + CHUNK_SIZE;
      byte[] blob = new byte[blobSize];
      rand.nextBytes(blob);
      Chunker chunker = new Chunker(blob, CHUNK_SIZE, DIGEST_UTIL);
      builders.add(chunker);
      blobsByHash.put(chunker.digest().getHash(), blob);
    }

    Set<String> uploadsFailedOnce = Collections.synchronizedSet(new HashSet<>());

    serviceRegistry.addService(new ByteStreamImplBase() {
      @Override
      public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> response) {
        return new StreamObserver<WriteRequest>() {

          private String digestHash;
          private byte[] receivedData;
          private long nextOffset;

          @Override
          public void onNext(WriteRequest writeRequest) {
            if (nextOffset == 0) {
              String resourceName = writeRequest.getResourceName();
              assertThat(resourceName).isNotEmpty();

              String[] components = resourceName.split("/");
              assertThat(components).hasLength(6);
              digestHash = components[4];
              assertThat(blobsByHash).containsKey(digestHash);
              receivedData = new byte[Integer.parseInt(components[5])];
            }
            assertThat(digestHash).isNotNull();
            // An upload for a given blob has a 10% chance to fail once during its lifetime.
            // This is to exercise the retry mechanism a bit.
            boolean shouldFail =
                rand.nextInt(10) == 0 && !uploadsFailedOnce.contains(digestHash);
            if (shouldFail) {
              uploadsFailedOnce.add(digestHash);
              response.onError(Status.INTERNAL.asException());
              return;
            }

            ByteString data = writeRequest.getData();
            System.arraycopy(
                data.toByteArray(), 0, receivedData, (int) nextOffset, data.size());
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
    });

    uploader.uploadBlobs(builders);

    blockUntilInternalStateConsistent(uploader);

    withEmptyMetadata.detach(prevContext);
  }

  @Test(timeout = 20000)
  public void contextShouldBePreservedUponRetries() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    // We upload blobs with different context, and retry 3 times for each upload.
    // We verify that the correct metadata is passed to the server with every blob.
    RemoteRetrier retrier =
        new RemoteRetrier(() -> new FixedBackoff(3, 0), (e) -> true, Retrier.ALLOW_ALL_CALLS);
    ByteStreamUploader uploader =
        new ByteStreamUploader(INSTANCE_NAME, channel, null, 3, retrier, retryService);

    List<String> toUpload = ImmutableList.of("aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc");
    List<Chunker> builders = new ArrayList<>(toUpload.size());
    Map<String, Integer> uploadsFailed = new HashMap<>();
    for (String s : toUpload) {
      Chunker chunker = new Chunker(s.getBytes(UTF_8), /* chunkSize=*/ 3, DIGEST_UTIL);
      builders.add(chunker);
      uploadsFailed.put(chunker.digest().getHash(), 0);
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
        };
    serviceRegistry.addService(
        ServerInterceptors.intercept(
            bsService, new TracingMetadataUtils.ServerHeadersInterceptor()));

    List<ListenableFuture<Void>> uploads = new ArrayList<>();

    for (Chunker chunker : builders) {
      Context ctx =
          TracingMetadataUtils.contextWithMetadata(
              "build-req-id", "command-id", DIGEST_UTIL.asActionKey(chunker.digest()));
      ctx.call(
          () -> {
            uploads.add(uploader.uploadBlobAsync(chunker));
            return null;
          });
    }

    for (ListenableFuture<Void> upload : uploads) {
      upload.get();
    }

    blockUntilInternalStateConsistent(uploader);

    withEmptyMetadata.detach(prevContext);
  }

  @Test(timeout = 10000)
  public void sameBlobShouldNotBeUploadedTwice() throws Exception {
    // Test that uploading the same file concurrently triggers only one file upload.

    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        new RemoteRetrier(() -> mockBackoff, (e) -> true, Retrier.ALLOW_ALL_CALLS);
    ByteStreamUploader uploader =
        new ByteStreamUploader(INSTANCE_NAME, channel, null, 3, retrier, retryService);

    byte[] blob = new byte[CHUNK_SIZE * 10];
    Chunker chunker = new Chunker(blob, CHUNK_SIZE, DIGEST_UTIL);

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

    Future<?> upload1 = uploader.uploadBlobAsync(chunker);
    Future<?> upload2 = uploader.uploadBlobAsync(chunker);

    blocker.countDown();

    assertThat(upload1).isSameAs(upload2);

    upload1.get();

    assertThat(numWriteCalls.get()).isEqualTo(1);

    withEmptyMetadata.detach(prevContext);
  }

  @Test(timeout = 10000)
  public void errorsShouldBeReported() throws IOException, InterruptedException {
    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        new RemoteRetrier(() -> new FixedBackoff(1, 10), (e) -> true, Retrier.ALLOW_ALL_CALLS);
    ByteStreamUploader uploader =
        new ByteStreamUploader(INSTANCE_NAME, channel, null, 3, retrier, retryService);

    byte[] blob = new byte[CHUNK_SIZE];
    Chunker chunker = new Chunker(blob, CHUNK_SIZE, DIGEST_UTIL);

    serviceRegistry.addService(new ByteStreamImplBase() {
      @Override
      public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> response) {
        response.onError(Status.INTERNAL.asException());
        return new NoopStreamObserver();
      }
    });

    try {
      uploader.uploadBlob(chunker);
      fail("Should have thrown an exception.");
    } catch (RetryException e) {
      assertThat(e.getAttempts()).isEqualTo(2);
      assertThat(RemoteRetrierUtils.causedByStatus(e, Code.INTERNAL)).isTrue();
    }

    withEmptyMetadata.detach(prevContext);
  }

  @Test(timeout = 10000)
  public void shutdownShouldCancelOngoingUploads() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        new RemoteRetrier(() -> new FixedBackoff(1, 10), (e) -> true, Retrier.ALLOW_ALL_CALLS);
    ByteStreamUploader uploader =
        new ByteStreamUploader(INSTANCE_NAME, channel, null, 3, retrier, retryService);

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
    Chunker chunker1 = new Chunker(blob1, CHUNK_SIZE, DIGEST_UTIL);

    byte[] blob2 = new byte[CHUNK_SIZE + 1];
    Chunker chunker2 = new Chunker(blob2, CHUNK_SIZE, DIGEST_UTIL);

    ListenableFuture<Void> f1 = uploader.uploadBlobAsync(chunker1);
    ListenableFuture<Void> f2 = uploader.uploadBlobAsync(chunker2);

    assertThat(uploader.uploadsInProgress()).isTrue();

    uploader.shutdown();

    cancellations.await();

    assertThat(f1.isCancelled()).isTrue();
    assertThat(f2.isCancelled()).isTrue();

    blockUntilInternalStateConsistent(uploader);

    withEmptyMetadata.detach(prevContext);
  }

  @Test(timeout = 10000)
  public void failureInRetryExecutorShouldBeHandled() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        new RemoteRetrier(() -> new FixedBackoff(1, 10), (e) -> true, Retrier.ALLOW_ALL_CALLS);
    ByteStreamUploader uploader =
        new ByteStreamUploader(INSTANCE_NAME, channel, null, 3, retrier, retryService);

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
    Chunker chunker = new Chunker(blob, CHUNK_SIZE, DIGEST_UTIL);
    try {
      uploader.uploadBlob(chunker);
      fail("Should have thrown an exception.");
    } catch (RetryException e) {
      assertThat(e).hasCauseThat().isInstanceOf(RejectedExecutionException.class);
    }

    withEmptyMetadata.detach(prevContext);
  }

  @Test(timeout = 10000)
  public void resourceNameWithoutInstanceName() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        new RemoteRetrier(() -> mockBackoff, (e) -> true, Retrier.ALLOW_ALL_CALLS);
    ByteStreamUploader uploader =
        new ByteStreamUploader(/* instanceName */ null, channel, null, 3, retrier, retryService);

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
    Chunker chunker = new Chunker(blob, CHUNK_SIZE, DIGEST_UTIL);

    uploader.uploadBlob(chunker);

    withEmptyMetadata.detach(prevContext);
  }

  @Test(timeout = 10000)
  public void nonRetryableStatusShouldNotBeRetried() throws Exception {
    Context prevContext = withEmptyMetadata.attach();
    RemoteRetrier retrier =
        new RemoteRetrier(
            () -> new FixedBackoff(1, 0),
            /* No Status is retriable. */ (e) -> false,
            Retrier.ALLOW_ALL_CALLS);
    ByteStreamUploader uploader =
        new ByteStreamUploader(/* instanceName */ null, channel, null, 3, retrier, retryService);

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
    Chunker chunker = new Chunker(blob, CHUNK_SIZE, DIGEST_UTIL);

    try {
      uploader.uploadBlob(chunker);
      fail("Should have thrown an exception.");
    } catch (RetryException e) {
      assertThat(numCalls.get()).isEqualTo(1);
    }

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

  private static class FixedBackoff implements Retrier.Backoff {

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

  private void blockUntilInternalStateConsistent(ByteStreamUploader uploader) throws Exception {
    // Poll until all upload futures have been removed from the internal hash map. The polling is
    // necessary, as listeners are executed after Future.get() calls are notified about completion.
    while (uploader.uploadsInProgress()) {
      Thread.sleep(1);
    }
  }
}

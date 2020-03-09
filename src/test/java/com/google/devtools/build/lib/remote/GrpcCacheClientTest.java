// Copyright 2015 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.fail;
import static org.mockito.AdditionalAnswers.answerVoid;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionCacheGrpc.ActionCacheImplBase;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.ContentAddressableStorageGrpc.ContentAddressableStorageImplBase;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.FindMissingBlobsRequest;
import build.bazel.remote.execution.v2.FindMissingBlobsResponse;
import build.bazel.remote.execution.v2.GetActionResultRequest;
import build.bazel.remote.execution.v2.Tree;
import build.bazel.remote.execution.v2.UpdateActionResultRequest;
import com.google.api.client.json.GenericJson;
import com.google.api.client.json.jackson2.JacksonFactory;
import com.google.bytestream.ByteStreamGrpc.ByteStreamImplBase;
import com.google.bytestream.ByteStreamProto.QueryWriteStatusRequest;
import com.google.bytestream.ByteStreamProto.QueryWriteStatusResponse;
import com.google.bytestream.ByteStreamProto.ReadRequest;
import com.google.bytestream.ByteStreamProto.ReadResponse;
import com.google.bytestream.ByteStreamProto.WriteRequest;
import com.google.bytestream.ByteStreamProto.WriteResponse;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Maps;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.authandtls.GoogleAuthUtils;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.RemoteRetrier.ExponentialBackoff;
import com.google.devtools.build.lib.remote.Retrier.Backoff;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.StringActionInput;
import com.google.devtools.build.lib.remote.util.TestUtils;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.protobuf.ByteString;
import io.grpc.BindableService;
import io.grpc.CallCredentials;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptor;
import io.grpc.Context;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.Server;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.ServerInterceptors;
import io.grpc.Status;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import io.grpc.util.MutableHandlerRegistry;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentMatchers;
import org.mockito.Mockito;
import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;

/** Tests for {@link GrpcCacheClient}. */
@RunWith(JUnit4.class)
public class GrpcCacheClientTest {

  private static final DigestUtil DIGEST_UTIL = new DigestUtil(DigestHashFunction.SHA256);

  private FileSystem fs;
  private Path execRoot;
  private FileOutErr outErr;
  private FakeActionInputFileCache fakeFileCache;
  private final MutableHandlerRegistry serviceRegistry = new MutableHandlerRegistry();
  private final String fakeServerName = "fake server for " + getClass();
  private Server fakeServer;
  private Context withEmptyMetadata;
  private Context prevContext;
  private ListeningScheduledExecutorService retryService;

  @Before
  public final void setUp() throws Exception {
    // Use a mutable service registry for later registering the service impl for each test case.
    fakeServer =
        InProcessServerBuilder.forName(fakeServerName)
            .fallbackHandlerRegistry(serviceRegistry)
            .directExecutor()
            .build()
            .start();
    Chunker.setDefaultChunkSizeForTesting(1000); // Enough for everything to be one chunk.
    fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    execRoot = fs.getPath("/exec/root");
    FileSystemUtils.createDirectoryAndParents(execRoot);
    fakeFileCache = new FakeActionInputFileCache(execRoot);

    Path stdout = fs.getPath("/tmp/stdout");
    Path stderr = fs.getPath("/tmp/stderr");
    FileSystemUtils.createDirectoryAndParents(stdout.getParentDirectory());
    FileSystemUtils.createDirectoryAndParents(stderr.getParentDirectory());
    outErr = new FileOutErr(stdout, stderr);
    withEmptyMetadata =
        TracingMetadataUtils.contextWithMetadata(
            "none", "none", DIGEST_UTIL.asActionKey(Digest.getDefaultInstance()));
    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));

    prevContext = withEmptyMetadata.attach();
  }

  @After
  public void tearDown() throws Exception {
    withEmptyMetadata.detach(prevContext);

    retryService.shutdownNow();
    retryService.awaitTermination(
        com.google.devtools.build.lib.testutil.TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS);

    fakeServer.shutdownNow();
    fakeServer.awaitTermination();
  }

  private static class CallCredentialsInterceptor implements ClientInterceptor {
    private final CallCredentials credentials;

    public CallCredentialsInterceptor(CallCredentials credentials) {
      this.credentials = credentials;
    }

    @Override
    public <RequestT, ResponseT> ClientCall<RequestT, ResponseT> interceptCall(
        MethodDescriptor<RequestT, ResponseT> method, CallOptions callOptions, Channel next) {
      assertThat(callOptions.getCredentials()).isEqualTo(credentials);
      // Remove the call credentials to allow testing with dummy ones.
      return next.newCall(method, callOptions.withCallCredentials(null));
    }
  }

  private GrpcCacheClient newClient() throws IOException {
    return newClient(Options.getDefaults(RemoteOptions.class));
  }

  private GrpcCacheClient newClient(RemoteOptions remoteOptions) throws IOException {
    return newClient(remoteOptions, () -> new ExponentialBackoff(remoteOptions));
  }

  private GrpcCacheClient newClient(RemoteOptions remoteOptions, Supplier<Backoff> backoffSupplier)
      throws IOException {
    AuthAndTLSOptions authTlsOptions = Options.getDefaults(AuthAndTLSOptions.class);
    authTlsOptions.useGoogleDefaultCredentials = true;
    authTlsOptions.googleCredentials = "/exec/root/creds.json";
    authTlsOptions.googleAuthScopes = ImmutableList.of("dummy.scope");

    GenericJson json = new GenericJson();
    json.put("type", "authorized_user");
    json.put("client_id", "some_client");
    json.put("client_secret", "foo");
    json.put("refresh_token", "bar");
    Scratch scratch = new Scratch();
    scratch.file(authTlsOptions.googleCredentials, new JacksonFactory().toString(json));

    CallCredentials creds;
    try (InputStream in = scratch.resolve(authTlsOptions.googleCredentials).getInputStream()) {
      creds = GoogleAuthUtils.newCallCredentials(in, authTlsOptions.googleAuthScopes);
    }

    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            backoffSupplier, RemoteRetrier.RETRIABLE_GRPC_ERRORS, retryService);
    ReferenceCountedChannel channel =
        new ReferenceCountedChannel(
            InProcessChannelBuilder.forName(fakeServerName)
                .directExecutor()
                .intercept(new CallCredentialsInterceptor(creds))
                .intercept(TracingMetadataUtils.newCacheHeadersInterceptor(remoteOptions))
                .build());
    ByteStreamUploader uploader =
        new ByteStreamUploader(
            remoteOptions.remoteInstanceName,
            channel.retain(),
            creds,
            remoteOptions.remoteTimeout,
            retrier);
    return new GrpcCacheClient(
        channel.retain(), creds, remoteOptions, retrier, DIGEST_UTIL, uploader);
  }

  private static byte[] downloadBlob(GrpcCacheClient cacheClient, Digest digest)
      throws IOException, InterruptedException {
    try (ByteArrayOutputStream out = new ByteArrayOutputStream()) {
      getFromFuture(cacheClient.downloadBlob(digest, out));
      return out.toByteArray();
    }
  }

  @Test
  public void testVirtualActionInputSupport() throws Exception {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    RemoteExecutionCache client =
        new RemoteExecutionCache(newClient(options), options, DIGEST_UTIL);
    PathFragment execPath = PathFragment.create("my/exec/path");
    VirtualActionInput virtualActionInput = new StringActionInput("hello", execPath);
    MerkleTree merkleTree =
        MerkleTree.build(
            ImmutableSortedMap.of(execPath, virtualActionInput),
            fakeFileCache,
            execRoot,
            DIGEST_UTIL);
    Digest digest = DIGEST_UTIL.compute(virtualActionInput.getBytes().toByteArray());

    // Add a fake CAS that responds saying that the above virtual action input is missing
    serviceRegistry.addService(
        new ContentAddressableStorageImplBase() {
          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            responseObserver.onNext(
                FindMissingBlobsResponse.newBuilder().addMissingBlobDigests(digest).build());
            responseObserver.onCompleted();
          }
        });

    // Mock a byte stream and assert that we see the virtual action input with contents 'hello'
    AtomicBoolean writeOccurred = new AtomicBoolean();
    serviceRegistry.addService(
        new ByteStreamImplBase() {
          @Override
          public StreamObserver<WriteRequest> write(
              final StreamObserver<WriteResponse> responseObserver) {
            return new StreamObserver<WriteRequest>() {
              @Override
              public void onNext(WriteRequest request) {
                assertThat(request.getResourceName()).contains(digest.getHash());
                assertThat(request.getFinishWrite()).isTrue();
                assertThat(request.getData().toStringUtf8()).isEqualTo("hello");
                writeOccurred.set(true);
              }

              @Override
              public void onCompleted() {
                responseObserver.onNext(WriteResponse.newBuilder().setCommittedSize(5).build());
                responseObserver.onCompleted();
              }

              @Override
              public void onError(Throwable t) {
                fail("An error occurred: " + t);
              }
            };
          }
        });

    // Upload all missing inputs (that is, the virtual action input from above)
    client.ensureInputsPresent(merkleTree, ImmutableMap.of());
  }

  @Test
  public void testDownloadEmptyBlob() throws Exception {
    GrpcCacheClient client = newClient();
    Digest emptyDigest = DIGEST_UTIL.compute(new byte[0]);
    // Will not call the mock Bytestream interface at all.
    assertThat(downloadBlob(client, emptyDigest)).isEmpty();
  }

  @Test
  public void testDownloadBlobSingleChunk() throws Exception {
    final GrpcCacheClient client = newClient();
    final Digest digest = DIGEST_UTIL.computeAsUtf8("abcdefg");
    serviceRegistry.addService(
        new ByteStreamImplBase() {
          @Override
          public void read(ReadRequest request, StreamObserver<ReadResponse> responseObserver) {
            assertThat(request.getResourceName().contains(digest.getHash())).isTrue();
            responseObserver.onNext(
                ReadResponse.newBuilder().setData(ByteString.copyFromUtf8("abcdefg")).build());
            responseObserver.onCompleted();
          }
        });
    assertThat(new String(downloadBlob(client, digest), UTF_8)).isEqualTo("abcdefg");
  }

  @Test
  public void testDownloadBlobMultipleChunks() throws Exception {
    final GrpcCacheClient client = newClient();
    final Digest digest = DIGEST_UTIL.computeAsUtf8("abcdefg");
    serviceRegistry.addService(
        new ByteStreamImplBase() {
          @Override
          public void read(ReadRequest request, StreamObserver<ReadResponse> responseObserver) {
            assertThat(request.getResourceName().contains(digest.getHash())).isTrue();
            responseObserver.onNext(
                ReadResponse.newBuilder().setData(ByteString.copyFromUtf8("abc")).build());
            responseObserver.onNext(
                ReadResponse.newBuilder().setData(ByteString.copyFromUtf8("def")).build());
            responseObserver.onNext(
                ReadResponse.newBuilder().setData(ByteString.copyFromUtf8("g")).build());
            responseObserver.onCompleted();
          }
        });
    assertThat(new String(downloadBlob(client, digest), UTF_8)).isEqualTo("abcdefg");
  }

  @Test
  public void testDownloadAllResults() throws Exception {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    GrpcCacheClient client = newClient(remoteOptions);
    RemoteCache remoteCache = new RemoteCache(client, remoteOptions, DIGEST_UTIL);

    Digest fooDigest = DIGEST_UTIL.computeAsUtf8("foo-contents");
    Digest barDigest = DIGEST_UTIL.computeAsUtf8("bar-contents");
    Digest emptyDigest = DIGEST_UTIL.compute(new byte[0]);
    serviceRegistry.addService(
        new FakeImmutableCacheByteStreamImpl(fooDigest, "foo-contents", barDigest, "bar-contents"));

    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputFilesBuilder().setPath("a/foo").setDigest(fooDigest);
    result.addOutputFilesBuilder().setPath("b/empty").setDigest(emptyDigest);
    result.addOutputFilesBuilder().setPath("a/bar").setDigest(barDigest).setIsExecutable(true);
    remoteCache.download(result.build(), execRoot, null, /* outputFilesLocker= */ () -> {});
    assertThat(DIGEST_UTIL.compute(execRoot.getRelative("a/foo"))).isEqualTo(fooDigest);
    assertThat(DIGEST_UTIL.compute(execRoot.getRelative("b/empty"))).isEqualTo(emptyDigest);
    assertThat(DIGEST_UTIL.compute(execRoot.getRelative("a/bar"))).isEqualTo(barDigest);
    assertThat(execRoot.getRelative("a/bar").isExecutable()).isTrue();
  }

  @Test
  public void testDownloadDirectory() throws Exception {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    GrpcCacheClient client = newClient(remoteOptions);
    RemoteCache remoteCache = new RemoteCache(client, remoteOptions, DIGEST_UTIL);

    Digest fooDigest = DIGEST_UTIL.computeAsUtf8("foo-contents");
    Digest quxDigest = DIGEST_UTIL.computeAsUtf8("qux-contents");
    Tree barTreeMessage =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addFiles(
                        FileNode.newBuilder()
                            .setName("qux")
                            .setDigest(quxDigest)
                            .setIsExecutable(true)))
            .build();
    Digest barTreeDigest = DIGEST_UTIL.compute(barTreeMessage);
    serviceRegistry.addService(
        new FakeImmutableCacheByteStreamImpl(
            ImmutableMap.of(
                fooDigest, "foo-contents",
                barTreeDigest, barTreeMessage.toByteString(),
                quxDigest, "qux-contents")));

    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputFilesBuilder().setPath("a/foo").setDigest(fooDigest);
    result.addOutputDirectoriesBuilder().setPath("a/bar").setTreeDigest(barTreeDigest);
    remoteCache.download(result.build(), execRoot, null, /* outputFilesLocker= */ () -> {});

    assertThat(DIGEST_UTIL.compute(execRoot.getRelative("a/foo"))).isEqualTo(fooDigest);
    assertThat(DIGEST_UTIL.compute(execRoot.getRelative("a/bar/qux"))).isEqualTo(quxDigest);
    assertThat(execRoot.getRelative("a/bar/qux").isExecutable()).isTrue();
  }

  @Test
  public void testDownloadDirectoryEmpty() throws Exception {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    GrpcCacheClient client = newClient(remoteOptions);
    RemoteCache remoteCache = new RemoteCache(client, remoteOptions, DIGEST_UTIL);

    Tree barTreeMessage = Tree.newBuilder().setRoot(Directory.newBuilder()).build();
    Digest barTreeDigest = DIGEST_UTIL.compute(barTreeMessage);
    serviceRegistry.addService(
        new FakeImmutableCacheByteStreamImpl(
            ImmutableMap.of(barTreeDigest, barTreeMessage.toByteString())));

    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputDirectoriesBuilder().setPath("a/bar").setTreeDigest(barTreeDigest);
    remoteCache.download(result.build(), execRoot, null, /* outputFilesLocker= */ () -> {});

    assertThat(execRoot.getRelative("a/bar").isDirectory()).isTrue();
  }

  @Test
  public void testDownloadDirectoryNested() throws Exception {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    GrpcCacheClient client = newClient(remoteOptions);
    RemoteCache remoteCache = new RemoteCache(client, remoteOptions, DIGEST_UTIL);

    Digest fooDigest = DIGEST_UTIL.computeAsUtf8("foo-contents");
    Digest quxDigest = DIGEST_UTIL.computeAsUtf8("qux-contents");
    Directory wobbleDirMessage =
        Directory.newBuilder()
            .addFiles(FileNode.newBuilder().setName("qux").setDigest(quxDigest))
            .build();
    Digest wobbleDirDigest = DIGEST_UTIL.compute(wobbleDirMessage);
    Tree barTreeMessage =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addFiles(
                        FileNode.newBuilder()
                            .setName("qux")
                            .setDigest(quxDigest)
                            .setIsExecutable(true))
                    .addDirectories(
                        DirectoryNode.newBuilder().setName("wobble").setDigest(wobbleDirDigest)))
            .addChildren(wobbleDirMessage)
            .build();
    Digest barTreeDigest = DIGEST_UTIL.compute(barTreeMessage);
    serviceRegistry.addService(
        new FakeImmutableCacheByteStreamImpl(
            ImmutableMap.of(
                fooDigest, "foo-contents",
                barTreeDigest, barTreeMessage.toByteString(),
                quxDigest, "qux-contents")));

    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputFilesBuilder().setPath("a/foo").setDigest(fooDigest);
    result.addOutputDirectoriesBuilder().setPath("a/bar").setTreeDigest(barTreeDigest);
    remoteCache.download(result.build(), execRoot, null, /* outputFilesLocker= */ () -> {});

    assertThat(DIGEST_UTIL.compute(execRoot.getRelative("a/foo"))).isEqualTo(fooDigest);
    assertThat(DIGEST_UTIL.compute(execRoot.getRelative("a/bar/wobble/qux"))).isEqualTo(quxDigest);
    assertThat(execRoot.getRelative("a/bar/wobble/qux").isExecutable()).isFalse();
  }

  static class TestChunkedRequestObserver implements StreamObserver<WriteRequest> {
    private final StreamObserver<WriteResponse> responseObserver;
    private final String contents;
    private final Chunker chunker;
    private final Digest digest;

    public TestChunkedRequestObserver(
        StreamObserver<WriteResponse> responseObserver, String contents, int chunkSizeBytes) {
      this.responseObserver = responseObserver;
      this.contents = contents;
      byte[] blob = contents.getBytes(UTF_8);
      chunker = Chunker.builder().setInput(blob).setChunkSize(chunkSizeBytes).build();
      digest = DIGEST_UTIL.compute(blob);
    }

    @Override
    public void onNext(WriteRequest request) {
      assertThat(chunker.hasNext()).isTrue();
      try {
        Chunker.Chunk chunk = chunker.next();
        long offset = chunk.getOffset();
        ByteString data = chunk.getData();
        if (offset == 0) {
          assertThat(request.getResourceName()).contains(digest.getHash());
        } else {
          assertThat(request.getResourceName()).isEmpty();
        }
        assertThat(request.getFinishWrite())
            .isEqualTo(offset + data.size() == digest.getSizeBytes());
        assertThat(request.getData()).isEqualTo(data);
      } catch (IOException e) {
        fail("An error occurred:" + e);
      }
    }

    @Override
    public void onCompleted() {
      assertThat(chunker.hasNext()).isFalse();
      responseObserver.onNext(
          WriteResponse.newBuilder().setCommittedSize(contents.length()).build());
      responseObserver.onCompleted();
    }

    @Override
    public void onError(Throwable t) {
      fail("An error occurred: " + t);
    }
  }

  @Test
  public void testUploadDirectory() throws Exception {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    GrpcCacheClient client = newClient(remoteOptions);
    RemoteCache remoteCache = new RemoteCache(client, remoteOptions, DIGEST_UTIL);

    final Digest fooDigest =
        fakeFileCache.createScratchInput(ActionInputHelper.fromPath("a/foo"), "xyz");
    final Digest quxDigest =
        fakeFileCache.createScratchInput(ActionInputHelper.fromPath("bar/qux"), "abc");
    final Digest barDigest =
        fakeFileCache.createScratchInputDirectory(
            ActionInputHelper.fromPath("bar"),
            Tree.newBuilder()
                .setRoot(
                    Directory.newBuilder()
                        .addFiles(
                            FileNode.newBuilder()
                                .setIsExecutable(true)
                                .setName("qux")
                                .setDigest(quxDigest)
                                .build())
                        .build())
                .build());
    final Path fooFile = execRoot.getRelative("a/foo");
    final Path quxFile = execRoot.getRelative("bar/qux");
    quxFile.setExecutable(true);
    final Path barDir = execRoot.getRelative("bar");
    serviceRegistry.addService(
        new ContentAddressableStorageImplBase() {
          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            assertThat(request.getBlobDigestsList())
                .containsAtLeast(fooDigest, quxDigest, barDigest);
            // Nothing is missing.
            responseObserver.onNext(FindMissingBlobsResponse.getDefaultInstance());
            responseObserver.onCompleted();
          }
        });
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void updateActionResult(
              UpdateActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onNext(request.getActionResult());
            responseObserver.onCompleted();
          }
        });

    ActionResult result = uploadDirectory(remoteCache, ImmutableList.<Path>of(fooFile, barDir));
    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputFilesBuilder().setPath("a/foo").setDigest(fooDigest);
    expectedResult.addOutputDirectoriesBuilder().setPath("bar").setTreeDigest(barDigest);
    assertThat(result).isEqualTo(expectedResult.build());
  }

  @Test
  public void testUploadDirectoryEmpty() throws Exception {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    GrpcCacheClient client = newClient(remoteOptions);
    RemoteCache remoteCache = new RemoteCache(client, remoteOptions, DIGEST_UTIL);

    final Digest barDigest =
        fakeFileCache.createScratchInputDirectory(
            ActionInputHelper.fromPath("bar"),
            Tree.newBuilder().setRoot(Directory.newBuilder().build()).build());
    final Path barDir = execRoot.getRelative("bar");
    serviceRegistry.addService(
        new ContentAddressableStorageImplBase() {
          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            assertThat(request.getBlobDigestsList()).contains(barDigest);
            // Nothing is missing.
            responseObserver.onNext(FindMissingBlobsResponse.getDefaultInstance());
            responseObserver.onCompleted();
          }
        });
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void updateActionResult(
              UpdateActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onNext(request.getActionResult());
            responseObserver.onCompleted();
          }
        });

    ActionResult result = uploadDirectory(remoteCache, ImmutableList.<Path>of(barDir));
    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectoriesBuilder().setPath("bar").setTreeDigest(barDigest);
    assertThat(result).isEqualTo(expectedResult.build());
  }

  @Test
  public void testUploadDirectoryNested() throws Exception {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    GrpcCacheClient client = newClient(remoteOptions);
    RemoteCache remoteCache = new RemoteCache(client, remoteOptions, DIGEST_UTIL);

    final Digest wobbleDigest =
        fakeFileCache.createScratchInput(ActionInputHelper.fromPath("bar/test/wobble"), "xyz");
    final Digest quxDigest =
        fakeFileCache.createScratchInput(ActionInputHelper.fromPath("bar/qux"), "abc");
    final Directory testDirMessage =
        Directory.newBuilder()
            .addFiles(FileNode.newBuilder().setName("wobble").setDigest(wobbleDigest).build())
            .build();
    final Digest testDigest = DIGEST_UTIL.compute(testDirMessage);
    final Tree barTree =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addFiles(
                        FileNode.newBuilder()
                            .setIsExecutable(true)
                            .setName("qux")
                            .setDigest(quxDigest))
                    .addDirectories(
                        DirectoryNode.newBuilder().setName("test").setDigest(testDigest)))
            .addChildren(testDirMessage)
            .build();
    final Digest barDigest =
        fakeFileCache.createScratchInputDirectory(ActionInputHelper.fromPath("bar"), barTree);
    final Path quxFile = execRoot.getRelative("bar/qux");
    quxFile.setExecutable(true);
    final Path barDir = execRoot.getRelative("bar");
    serviceRegistry.addService(
        new ContentAddressableStorageImplBase() {
          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            assertThat(request.getBlobDigestsList())
                .containsAtLeast(quxDigest, barDigest, wobbleDigest);
            // Nothing is missing.
            responseObserver.onNext(FindMissingBlobsResponse.getDefaultInstance());
            responseObserver.onCompleted();
          }
        });
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void updateActionResult(
              UpdateActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onNext(request.getActionResult());
            responseObserver.onCompleted();
          }
        });

    ActionResult result = uploadDirectory(remoteCache, ImmutableList.of(barDir));
    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectoriesBuilder().setPath("bar").setTreeDigest(barDigest);
    assertThat(result).isEqualTo(expectedResult.build());
  }

  private ActionResult uploadDirectory(RemoteCache remoteCache, List<Path> outputs)
      throws Exception {
    Action action = Action.getDefaultInstance();
    ActionKey actionKey = DIGEST_UTIL.computeActionKey(action);
    Command cmd = Command.getDefaultInstance();
    return remoteCache.upload(actionKey, action, cmd, execRoot, outputs, outErr);
  }

  @Test
  public void extraHeaders() throws Exception {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    remoteOptions.remoteHeaders =
        ImmutableList.of(
            Maps.immutableEntry("CommonKey1", "CommonValue1"),
            Maps.immutableEntry("CommonKey2", "CommonValue2"));
    remoteOptions.remoteExecHeaders =
        ImmutableList.of(
            Maps.immutableEntry("ExecKey1", "ExecValue1"),
            Maps.immutableEntry("ExecKey2", "ExecValue2"));
    remoteOptions.remoteCacheHeaders =
        ImmutableList.of(
            Maps.immutableEntry("CacheKey1", "CacheValue1"),
            Maps.immutableEntry("CacheKey2", "CacheValue2"));

    ServerInterceptor interceptor =
        new ServerInterceptor() {
          @Override
          public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(
              ServerCall<ReqT, RespT> call,
              Metadata metadata,
              ServerCallHandler<ReqT, RespT> next) {
            assertThat(
                    metadata.get(Metadata.Key.of("CommonKey1", Metadata.ASCII_STRING_MARSHALLER)))
                .isEqualTo("CommonValue1");
            assertThat(
                    metadata.get(Metadata.Key.of("CommonKey2", Metadata.ASCII_STRING_MARSHALLER)))
                .isEqualTo("CommonValue2");
            assertThat(metadata.get(Metadata.Key.of("CacheKey1", Metadata.ASCII_STRING_MARSHALLER)))
                .isEqualTo("CacheValue1");
            assertThat(metadata.get(Metadata.Key.of("CacheKey2", Metadata.ASCII_STRING_MARSHALLER)))
                .isEqualTo("CacheValue2");
            assertThat(metadata.get(Metadata.Key.of("ExecKey1", Metadata.ASCII_STRING_MARSHALLER)))
                .isEqualTo(null);
            assertThat(metadata.get(Metadata.Key.of("ExecKey2", Metadata.ASCII_STRING_MARSHALLER)))
                .isEqualTo(null);
            return next.startCall(call, metadata);
          }
        };

    BindableService cas =
        new ContentAddressableStorageImplBase() {
          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            responseObserver.onNext(FindMissingBlobsResponse.getDefaultInstance());
            responseObserver.onCompleted();
          }
        };
    serviceRegistry.addService(cas);
    BindableService actionCache =
        new ActionCacheImplBase() {
          @Override
          public void getActionResult(
              GetActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onNext(ActionResult.getDefaultInstance());
            responseObserver.onCompleted();
          }
        };
    serviceRegistry.addService(ServerInterceptors.intercept(actionCache, interceptor));

    GrpcCacheClient client = newClient(remoteOptions);
    RemoteCache remoteCache = new RemoteCache(client, remoteOptions, DIGEST_UTIL);
    remoteCache.downloadActionResult(DIGEST_UTIL.asActionKey(DIGEST_UTIL.computeAsUtf8("key")));
  }

  @Test
  public void testUpload() throws Exception {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    GrpcCacheClient client = newClient(remoteOptions);
    RemoteCache remoteCache = new RemoteCache(client, remoteOptions, DIGEST_UTIL);

    final Digest fooDigest =
        fakeFileCache.createScratchInput(ActionInputHelper.fromPath("a/foo"), "xyz");
    final Digest barDigest =
        fakeFileCache.createScratchInput(ActionInputHelper.fromPath("bar"), "x");
    final Path fooFile = execRoot.getRelative("a/foo");
    final Path barFile = execRoot.getRelative("bar");
    barFile.setExecutable(true);
    Command command = Command.newBuilder().addOutputFiles("a/foo").build();
    final Digest cmdDigest = DIGEST_UTIL.compute(command.toByteArray());
    Action action = Action.newBuilder().setCommandDigest(cmdDigest).build();
    final Digest actionDigest = DIGEST_UTIL.compute(action.toByteArray());

    outErr.getOutputStream().write("foo out".getBytes(UTF_8));
    outErr.getOutputStream().close();
    outErr.getErrorStream().write("foo err".getBytes(UTF_8));
    outErr.getOutputStream().close();

    final Digest stdoutDigest = DIGEST_UTIL.compute(outErr.getOutputPath());
    final Digest stderrDigest = DIGEST_UTIL.compute(outErr.getErrorPath());

    serviceRegistry.addService(
        new ContentAddressableStorageImplBase() {
          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            assertThat(request.getBlobDigestsList())
                .containsExactly(
                    cmdDigest, actionDigest, fooDigest, barDigest, stdoutDigest, stderrDigest);
            // Nothing is missing.
            responseObserver.onNext(FindMissingBlobsResponse.getDefaultInstance());
            responseObserver.onCompleted();
          }
        });
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void updateActionResult(
              UpdateActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onNext(request.getActionResult());
            responseObserver.onCompleted();
          }
        });

    ActionResult result =
        remoteCache.upload(
            DIGEST_UTIL.asActionKey(actionDigest),
            action,
            command,
            execRoot,
            ImmutableList.of(fooFile, barFile),
            outErr);
    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.setStdoutDigest(stdoutDigest);
    expectedResult.setStderrDigest(stderrDigest);
    expectedResult.addOutputFilesBuilder().setPath("a/foo").setDigest(fooDigest);
    expectedResult
        .addOutputFilesBuilder()
        .setPath("bar")
        .setDigest(barDigest)
        .setIsExecutable(true);
    assertThat(result).isEqualTo(expectedResult.build());
  }

  @Test
  public void testUploadSplitMissingDigestsCall() throws Exception {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    remoteOptions.maxOutboundMessageSize = 80; // Enough for one digest, but not two.
    GrpcCacheClient client = newClient(remoteOptions);
    RemoteCache remoteCache = new RemoteCache(client, remoteOptions, DIGEST_UTIL);

    final Digest fooDigest =
        fakeFileCache.createScratchInput(ActionInputHelper.fromPath("a/foo"), "xyz");
    final Digest barDigest =
        fakeFileCache.createScratchInput(ActionInputHelper.fromPath("bar"), "x");
    final Path fooFile = execRoot.getRelative("a/foo");
    final Path barFile = execRoot.getRelative("bar");
    barFile.setExecutable(true);
    Command command = Command.newBuilder().addOutputFiles("a/foo").build();
    final Digest cmdDigest = DIGEST_UTIL.compute(command.toByteArray());
    Action action = Action.newBuilder().setCommandDigest(cmdDigest).build();
    final Digest actionDigest = DIGEST_UTIL.compute(action.toByteArray());
    AtomicInteger numGetMissingCalls = new AtomicInteger();
    serviceRegistry.addService(
        new ContentAddressableStorageImplBase() {
          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            numGetMissingCalls.incrementAndGet();
            assertThat(request.getBlobDigestsCount()).isEqualTo(1);
            // Nothing is missing.
            responseObserver.onNext(FindMissingBlobsResponse.getDefaultInstance());
            responseObserver.onCompleted();
          }
        });
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void updateActionResult(
              UpdateActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onNext(request.getActionResult());
            responseObserver.onCompleted();
          }
        });

    ActionResult result =
        remoteCache.upload(
            DIGEST_UTIL.asActionKey(actionDigest),
            action,
            command,
            execRoot,
            ImmutableList.of(fooFile, barFile),
            outErr);
    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputFilesBuilder().setPath("a/foo").setDigest(fooDigest);
    expectedResult
        .addOutputFilesBuilder()
        .setPath("bar")
        .setDigest(barDigest)
        .setIsExecutable(true);
    assertThat(result).isEqualTo(expectedResult.build());
    assertThat(numGetMissingCalls.get()).isEqualTo(4);
  }

  @Test
  public void testUploadCacheMissesWithRetries() throws Exception {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    GrpcCacheClient client = newClient(remoteOptions);
    RemoteCache remoteCache = new RemoteCache(client, remoteOptions, DIGEST_UTIL);

    final Digest fooDigest =
        fakeFileCache.createScratchInput(ActionInputHelper.fromPath("a/foo"), "xyz");
    final Digest barDigest =
        fakeFileCache.createScratchInput(ActionInputHelper.fromPath("bar"), "x");
    final Digest bazDigest =
        fakeFileCache.createScratchInput(ActionInputHelper.fromPath("baz"), "z");
    final Path fooFile = execRoot.getRelative("a/foo");
    final Path barFile = execRoot.getRelative("bar");
    final Path bazFile = execRoot.getRelative("baz");
    ActionKey actionKey = DIGEST_UTIL.asActionKey(fooDigest); // Could be any key.
    barFile.setExecutable(true);
    serviceRegistry.addService(
        new ContentAddressableStorageImplBase() {
          private int numErrors = 4;

          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            if (numErrors-- <= 0) {
              // All outputs are missing.
              responseObserver.onNext(
                  FindMissingBlobsResponse.newBuilder()
                      .addMissingBlobDigests(fooDigest)
                      .addMissingBlobDigests(barDigest)
                      .addMissingBlobDigests(bazDigest)
                      .build());
              responseObserver.onCompleted();
            } else {
              responseObserver.onError(Status.UNAVAILABLE.asRuntimeException());
            }
          }
        });
    ActionResult.Builder rb = ActionResult.newBuilder();
    rb.addOutputFilesBuilder().setPath("a/foo").setDigest(fooDigest);
    rb.addOutputFilesBuilder().setPath("bar").setDigest(barDigest).setIsExecutable(true);
    rb.addOutputFilesBuilder().setPath("baz").setDigest(bazDigest);
    ActionResult result = rb.build();
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          private int numErrors = 4;

          @Override
          public void updateActionResult(
              UpdateActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            assertThat(request)
                .isEqualTo(
                    UpdateActionResultRequest.newBuilder()
                        .setActionDigest(fooDigest)
                        .setActionResult(result)
                        .build());
            if (numErrors-- <= 0) {
              responseObserver.onNext(result);
              responseObserver.onCompleted();
            } else {
              responseObserver.onError(Status.UNAVAILABLE.asRuntimeException());
            }
          }
        });
    ByteStreamImplBase mockByteStreamImpl = Mockito.mock(ByteStreamImplBase.class);
    serviceRegistry.addService(mockByteStreamImpl);
    when(mockByteStreamImpl.write(ArgumentMatchers.<StreamObserver<WriteResponse>>any()))
        .thenAnswer(
            new Answer<StreamObserver<WriteRequest>>() {
              private int numErrors = 4;

              @Override
              @SuppressWarnings("unchecked")
              public StreamObserver<WriteRequest> answer(InvocationOnMock invocation) {
                StreamObserver<WriteResponse> responseObserver =
                    (StreamObserver<WriteResponse>) invocation.getArguments()[0];
                return new StreamObserver<WriteRequest>() {
                  @Override
                  public void onNext(WriteRequest request) {
                    numErrors--;
                    if (numErrors >= 0) {
                      responseObserver.onError(Status.UNAVAILABLE.asRuntimeException());
                      return;
                    }
                    assertThat(request.getFinishWrite()).isTrue();
                    String resourceName = request.getResourceName();
                    String dataStr = request.getData().toStringUtf8();
                    int size = 0;
                    if (resourceName.contains(fooDigest.getHash())) {
                      assertThat(dataStr).isEqualTo("xyz");
                      size = 3;
                    } else if (resourceName.contains(barDigest.getHash())) {
                      assertThat(dataStr).isEqualTo("x");
                      size = 1;
                    } else if (resourceName.contains(bazDigest.getHash())) {
                      assertThat(dataStr).isEqualTo("z");
                      size = 1;
                    } else {
                      fail("Unexpected resource name in upload: " + resourceName);
                    }
                    responseObserver.onNext(
                        WriteResponse.newBuilder().setCommittedSize(size).build());
                  }

                  @Override
                  public void onCompleted() {
                    responseObserver.onCompleted();
                  }

                  @Override
                  public void onError(Throwable t) {
                    fail("An error occurred: " + t);
                  }
                };
              }
            });
    doAnswer(
            answerVoid(
                (QueryWriteStatusRequest request,
                    StreamObserver<QueryWriteStatusResponse> responseObserver) -> {
                  responseObserver.onNext(
                      QueryWriteStatusResponse.newBuilder()
                          .setCommittedSize(0)
                          .setComplete(false)
                          .build());
                  responseObserver.onCompleted();
                }))
        .when(mockByteStreamImpl)
        .queryWriteStatus(any(), any());
    remoteCache.upload(
        actionKey,
        Action.getDefaultInstance(),
        Command.getDefaultInstance(),
        execRoot,
        ImmutableList.<Path>of(fooFile, barFile, bazFile),
        outErr);
    // 4 times for the errors, 3 times for the successful uploads.
    Mockito.verify(mockByteStreamImpl, Mockito.times(7))
        .write(ArgumentMatchers.<StreamObserver<WriteResponse>>any());
  }

  @Test
  public void testGetCachedActionResultWithRetries() throws Exception {
    final GrpcCacheClient client = newClient();
    ActionKey actionKey = DIGEST_UTIL.asActionKey(DIGEST_UTIL.computeAsUtf8("key"));
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          private int numErrors = 4;

          @Override
          public void getActionResult(
              GetActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onError(
                (numErrors-- <= 0 ? Status.NOT_FOUND : Status.UNAVAILABLE).asRuntimeException());
          }
        });
    assertThat(getFromFuture(client.downloadActionResult(actionKey))).isNull();
  }

  @Test
  public void downloadBlobIsRetriedWithProgress() throws IOException, InterruptedException {
    Backoff mockBackoff = Mockito.mock(Backoff.class);
    final GrpcCacheClient client =
        newClient(Options.getDefaults(RemoteOptions.class), () -> mockBackoff);
    final Digest digest = DIGEST_UTIL.computeAsUtf8("abcdefg");
    serviceRegistry.addService(
        new ByteStreamImplBase() {
          @Override
          public void read(ReadRequest request, StreamObserver<ReadResponse> responseObserver) {
            assertThat(request.getResourceName().contains(digest.getHash())).isTrue();
            ByteString data = ByteString.copyFromUtf8("abcdefg");
            int off = (int) request.getReadOffset();
            if (off == 0) {
              data = data.substring(0, 1);
            } else {
              data = data.substring(off);
            }
            responseObserver.onNext(ReadResponse.newBuilder().setData(data).build());
            if (off == 0) {
              responseObserver.onError(Status.DEADLINE_EXCEEDED.asException());
            } else {
              responseObserver.onCompleted();
            }
          }
        });
    assertThat(new String(downloadBlob(client, digest), UTF_8)).isEqualTo("abcdefg");
    Mockito.verify(mockBackoff, Mockito.never()).nextDelayMillis();
  }

  @Test
  public void downloadBlobPassesThroughDeadlineExceededWithoutProgress() throws IOException {
    Backoff mockBackoff = Mockito.mock(Backoff.class);
    Mockito.when(mockBackoff.nextDelayMillis()).thenReturn(-1L);
    final GrpcCacheClient client =
        newClient(Options.getDefaults(RemoteOptions.class), () -> mockBackoff);
    final Digest digest = DIGEST_UTIL.computeAsUtf8("abcdefg");
    serviceRegistry.addService(
        new ByteStreamImplBase() {
          @Override
          public void read(ReadRequest request, StreamObserver<ReadResponse> responseObserver) {
            assertThat(request.getResourceName().contains(digest.getHash())).isTrue();
            ByteString data = ByteString.copyFromUtf8("abcdefg");
            if (request.getReadOffset() == 0) {
              responseObserver.onNext(
                  ReadResponse.newBuilder().setData(data.substring(0, 2)).build());
            }
            responseObserver.onError(Status.DEADLINE_EXCEEDED.asException());
          }
        });
    IOException e = assertThrows(IOException.class, () -> downloadBlob(client, digest));
    Status st = Status.fromThrowable(e);
    assertThat(st.getCode()).isEqualTo(Status.Code.DEADLINE_EXCEEDED);
    Mockito.verify(mockBackoff, Mockito.times(1)).nextDelayMillis();
  }

  @Test
  public void testDownloadFailsOnDigestMismatch() throws Exception {
    // Test that the download fails when a blob/file has a different content hash than expected.

    GrpcCacheClient client = newClient();
    Digest digest = DIGEST_UTIL.computeAsUtf8("foo");
    serviceRegistry.addService(
        new ByteStreamImplBase() {
          @Override
          public void read(ReadRequest request, StreamObserver<ReadResponse> responseObserver) {
            ByteString data = ByteString.copyFromUtf8("bar");
            responseObserver.onNext(ReadResponse.newBuilder().setData(data).build());
            responseObserver.onCompleted();
          }
        });
    IOException e = assertThrows(IOException.class, () -> downloadBlob(client, digest));
    assertThat(e).hasMessageThat().contains(digest.getHash());
    assertThat(e).hasMessageThat().contains(DIGEST_UTIL.computeAsUtf8("bar").getHash());
  }

  @Test
  public void testDisablingDigestVerification() throws Exception {
    // Test that when digest verification is disabled a corrupted download works.

    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    remoteOptions.remoteVerifyDownloads = false;

    GrpcCacheClient client = newClient(remoteOptions);
    Digest digest = DIGEST_UTIL.computeAsUtf8("foo");
    ByteString downloadContents = ByteString.copyFromUtf8("bar");
    serviceRegistry.addService(
        new ByteStreamImplBase() {
          @Override
          public void read(ReadRequest request, StreamObserver<ReadResponse> responseObserver) {
            responseObserver.onNext(ReadResponse.newBuilder().setData(downloadContents).build());
            responseObserver.onCompleted();
          }
        });

    assertThat(downloadBlob(client, digest)).isEqualTo(downloadContents.toByteArray());
  }

  @Test
  public void isRemoteCacheOptionsWhenGrpcEnabled() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "grpc://some-host.com";

    assertThat(GrpcCacheClient.isRemoteCacheOptions(options)).isTrue();
  }

  @Test
  public void isRemoteCacheOptionsWhenGrpcEnabledUpperCase() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "GRPC://some-host.com";

    assertThat(GrpcCacheClient.isRemoteCacheOptions(options)).isTrue();
  }

  @Test
  public void isRemoteCacheOptionsWhenDefaultRemoteCacheEnabledForLocalhost() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "localhost:1234";

    assertThat(GrpcCacheClient.isRemoteCacheOptions(options)).isTrue();
  }

  @Test
  public void isRemoteCacheOptionsWhenDefaultRemoteCacheEnabled() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "some-host.com:1234";

    assertThat(GrpcCacheClient.isRemoteCacheOptions(options)).isTrue();
  }

  @Test
  public void isRemoteCacheOptionsWhenHttpEnabled() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "http://some-host.com";

    assertThat(GrpcCacheClient.isRemoteCacheOptions(options)).isFalse();
  }

  @Test
  public void isRemoteCacheOptionsWhenHttpEnabledWithUpperCase() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "HTTP://some-host.com";

    assertThat(GrpcCacheClient.isRemoteCacheOptions(options)).isFalse();
  }

  @Test
  public void isRemoteCacheOptionsWhenHttpsEnabled() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "https://some-host.com";

    assertThat(GrpcCacheClient.isRemoteCacheOptions(options)).isFalse();
  }

  @Test
  public void isRemoteCacheOptionsWhenUnknownScheme() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "grp://some-host.com";

    // TODO(ishikhman): add proper vaildation and flip to false
    assertThat(GrpcCacheClient.isRemoteCacheOptions(options)).isTrue();
  }

  @Test
  public void isRemoteCacheOptionsWhenUnknownSchemeStartsAsGrpc() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "grpcsss://some-host.com";

    // TODO(ishikhman): add proper vaildation and flip to false
    assertThat(GrpcCacheClient.isRemoteCacheOptions(options)).isTrue();
  }

  @Test
  public void isRemoteCacheOptionsWhenEmptyCacheProvided() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "";

    assertThat(GrpcCacheClient.isRemoteCacheOptions(options)).isFalse();
  }

  @Test
  public void isRemoteCacheOptionsWhenRemoteCacheDisabled() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);

    assertThat(GrpcCacheClient.isRemoteCacheOptions(options)).isFalse();
  }
}

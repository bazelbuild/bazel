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
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.fail;
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
import com.google.bytestream.ByteStreamProto.ReadRequest;
import com.google.bytestream.ByteStreamProto.ReadResponse;
import com.google.bytestream.ByteStreamProto.WriteRequest;
import com.google.bytestream.ByteStreamProto.WriteResponse;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.authandtls.GoogleAuthUtils;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.DigestUtil.ActionKey;
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
import io.grpc.CallCredentials;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptor;
import io.grpc.Context;
import io.grpc.MethodDescriptor;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import io.grpc.util.MutableHandlerRegistry;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;
import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;

/** Tests for {@link GrpcRemoteCache}. */
@RunWith(JUnit4.class)
public class GrpcRemoteCacheTest {

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
  private static ListeningScheduledExecutorService retryService;

  @BeforeClass
  public static void beforeEverything() {
    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
  }

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
    prevContext = withEmptyMetadata.attach();
  }

  @After
  public void tearDown() throws Exception {
    withEmptyMetadata.detach(prevContext);
    fakeServer.shutdownNow();
    fakeServer.awaitTermination();
  }

  @AfterClass
  public static void afterEverything() {
    retryService.shutdownNow();
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

  private GrpcRemoteCache newClient() throws IOException {
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
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    RemoteRetrier retrier =
        new RemoteRetrier(
            remoteOptions,
            RemoteRetrier.RETRIABLE_GRPC_ERRORS,
            retryService,
            Retrier.ALLOW_ALL_CALLS);
    ReferenceCountedChannel channel =
        new ReferenceCountedChannel(InProcessChannelBuilder.forName(fakeServerName).directExecutor()
            .intercept(new CallCredentialsInterceptor(creds)).build());
    ByteStreamUploader uploader =
        new ByteStreamUploader(remoteOptions.remoteInstanceName, channel.retain(), creds,
            remoteOptions.remoteTimeout, retrier);
    return new GrpcRemoteCache(channel.retain(),
        creds,
        remoteOptions,
        retrier,
        DIGEST_UTIL,
        uploader);
  }

  static class StringVirtualActionInput implements VirtualActionInput {
    private final String contents;
    private final PathFragment execPath;

    StringVirtualActionInput(String contents, PathFragment execPath) {
      this.contents = contents;
      this.execPath = execPath;
    }

    @Override
    public void writeTo(OutputStream out) throws IOException {
      out.write(contents.getBytes(StandardCharsets.UTF_8));
    }

    @Override
    public ByteString getBytes() throws IOException {
      ByteString.Output out = ByteString.newOutput();
      writeTo(out);
      return out.toByteString();
    }

    @Override
    public String getExecPathString() {
      return execPath.getPathString();
    }

    @Override
    public PathFragment getExecPath() {
      return execPath;
    }
  }

  @Test
  public void testVirtualActionInputSupport() throws Exception {
    GrpcRemoteCache client = newClient();
    TreeNodeRepository treeNodeRepository =
        new TreeNodeRepository(execRoot, fakeFileCache, DIGEST_UTIL, true);
    PathFragment execPath = PathFragment.create("my/exec/path");
    VirtualActionInput virtualActionInput = new StringVirtualActionInput("hello", execPath);
    Digest digest = DIGEST_UTIL.compute(virtualActionInput.getBytes().toByteArray());
    TreeNode root =
        treeNodeRepository.buildFromActionInputs(
            ImmutableSortedMap.of(execPath, virtualActionInput));

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
    client.ensureInputsPresent(
        treeNodeRepository,
        execRoot,
        root,
        Action.getDefaultInstance(),
        Command.getDefaultInstance());
    assertThat(writeOccurred.get()).named("WriteOccurred").isTrue();
  }

  @Test
  public void testDownloadEmptyBlob() throws Exception {
    GrpcRemoteCache client = newClient();
    Digest emptyDigest = DIGEST_UTIL.compute(new byte[0]);
    // Will not call the mock Bytestream interface at all.
    assertThat(getFromFuture(client.downloadBlob(emptyDigest))).isEmpty();
  }

  @Test
  public void testDownloadBlobSingleChunk() throws Exception {
    final GrpcRemoteCache client = newClient();
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
    assertThat(new String(getFromFuture(client.downloadBlob(digest)), UTF_8)).isEqualTo("abcdefg");
  }

  @Test
  public void testDownloadBlobMultipleChunks() throws Exception {
    final GrpcRemoteCache client = newClient();
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
    assertThat(new String(getFromFuture(client.downloadBlob(digest)), UTF_8)).isEqualTo("abcdefg");
  }

  @Test
  public void testDownloadAllResults() throws Exception {
    GrpcRemoteCache client = newClient();
    Digest fooDigest = DIGEST_UTIL.computeAsUtf8("foo-contents");
    Digest barDigest = DIGEST_UTIL.computeAsUtf8("bar-contents");
    Digest emptyDigest = DIGEST_UTIL.compute(new byte[0]);
    serviceRegistry.addService(
        new FakeImmutableCacheByteStreamImpl(fooDigest, "foo-contents", barDigest, "bar-contents"));

    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputFilesBuilder().setPath("a/foo").setDigest(fooDigest);
    result.addOutputFilesBuilder().setPath("b/empty").setDigest(emptyDigest);
    result.addOutputFilesBuilder().setPath("a/bar").setDigest(barDigest).setIsExecutable(true);
    client.download(result.build(), execRoot, null);
    assertThat(DIGEST_UTIL.compute(execRoot.getRelative("a/foo"))).isEqualTo(fooDigest);
    assertThat(DIGEST_UTIL.compute(execRoot.getRelative("b/empty"))).isEqualTo(emptyDigest);
    assertThat(DIGEST_UTIL.compute(execRoot.getRelative("a/bar"))).isEqualTo(barDigest);
    assertThat(execRoot.getRelative("a/bar").isExecutable()).isTrue();
  }

  @Test
  public void testDownloadDirectory() throws Exception {
    GrpcRemoteCache client = newClient();
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
    client.download(result.build(), execRoot, null);

    assertThat(DIGEST_UTIL.compute(execRoot.getRelative("a/foo"))).isEqualTo(fooDigest);
    assertThat(DIGEST_UTIL.compute(execRoot.getRelative("a/bar/qux"))).isEqualTo(quxDigest);
    assertThat(execRoot.getRelative("a/bar/qux").isExecutable()).isTrue();
  }

  @Test
  public void testDownloadDirectoryEmpty() throws Exception {
    GrpcRemoteCache client = newClient();
    Tree barTreeMessage = Tree.newBuilder().setRoot(Directory.newBuilder()).build();
    Digest barTreeDigest = DIGEST_UTIL.compute(barTreeMessage);
    serviceRegistry.addService(
        new FakeImmutableCacheByteStreamImpl(
            ImmutableMap.of(barTreeDigest, barTreeMessage.toByteString())));

    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputDirectoriesBuilder().setPath("a/bar").setTreeDigest(barTreeDigest);
    client.download(result.build(), execRoot, null);

    assertThat(execRoot.getRelative("a/bar").isDirectory()).isTrue();
  }

  @Test
  public void testDownloadDirectoryNested() throws Exception {
    GrpcRemoteCache client = newClient();
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
    client.download(result.build(), execRoot, null);

    assertThat(DIGEST_UTIL.compute(execRoot.getRelative("a/foo"))).isEqualTo(fooDigest);
    assertThat(DIGEST_UTIL.compute(execRoot.getRelative("a/bar/wobble/qux"))).isEqualTo(quxDigest);
    assertThat(execRoot.getRelative("a/bar/wobble/qux").isExecutable()).isFalse();
  }

  @Test
  public void testUploadBlobCacheHitWithRetries() throws Exception {
    final GrpcRemoteCache client = newClient();
    final Digest digest = DIGEST_UTIL.computeAsUtf8("abcdefg");
    serviceRegistry.addService(
        new ContentAddressableStorageImplBase() {
          private int numErrors = 4;

          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            if (numErrors-- <= 0) {
              responseObserver.onNext(FindMissingBlobsResponse.getDefaultInstance());
              responseObserver.onCompleted();
            } else {
              responseObserver.onError(Status.UNAVAILABLE.asRuntimeException());
            }
          }
        });
    assertThat(client.uploadBlob("abcdefg".getBytes(UTF_8))).isEqualTo(digest);
  }

  @Test
  public void testUploadBlobSingleChunk() throws Exception {
    final GrpcRemoteCache client = newClient();
    final Digest digest = DIGEST_UTIL.computeAsUtf8("abcdefg");
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
                assertThat(request.getData().toStringUtf8()).isEqualTo("abcdefg");
              }

              @Override
              public void onCompleted() {
                responseObserver.onNext(WriteResponse.newBuilder().setCommittedSize(7).build());
                responseObserver.onCompleted();
              }

              @Override
              public void onError(Throwable t) {
                fail("An error occurred: " + t);
              }
            };
          }
        });
    assertThat(client.uploadBlob("abcdefg".getBytes(UTF_8))).isEqualTo(digest);
  }

  static class TestChunkedRequestObserver implements StreamObserver<WriteRequest> {
    private final StreamObserver<WriteResponse> responseObserver;
    private final String contents;
    private Chunker chunker;

    public TestChunkedRequestObserver(
        StreamObserver<WriteResponse> responseObserver, String contents, int chunkSizeBytes) {
      this.responseObserver = responseObserver;
      this.contents = contents;
      chunker =
          Chunker.builder(DIGEST_UTIL)
              .setInput(contents.getBytes(UTF_8))
              .setChunkSize(chunkSizeBytes)
              .build();
    }

    @Override
    public void onNext(WriteRequest request) {
      assertThat(chunker.hasNext()).isTrue();
      try {
        Chunker.Chunk chunk = chunker.next();
        Digest digest = chunk.getDigest();
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

  private Answer<StreamObserver<WriteRequest>> blobChunkedWriteAnswer(
      final String contents, final int chunkSize) {
    return new Answer<StreamObserver<WriteRequest>>() {
      @Override
      @SuppressWarnings("unchecked")
      public StreamObserver<WriteRequest> answer(InvocationOnMock invocation) {
        return new TestChunkedRequestObserver(
            (StreamObserver<WriteResponse>) invocation.getArguments()[0], contents, chunkSize);
      }
    };
  }

  @Test
  public void testUploadBlobMultipleChunks() throws Exception {
    final Digest digest = DIGEST_UTIL.computeAsUtf8("abcdef");
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

    ByteStreamImplBase mockByteStreamImpl = Mockito.mock(ByteStreamImplBase.class);
    serviceRegistry.addService(mockByteStreamImpl);
    for (int chunkSize = 1; chunkSize <= 6; ++chunkSize) {
      GrpcRemoteCache client = newClient();
      Chunker.setDefaultChunkSizeForTesting(chunkSize);
      when(mockByteStreamImpl.write(Mockito.<StreamObserver<WriteResponse>>anyObject()))
          .thenAnswer(blobChunkedWriteAnswer("abcdef", chunkSize));
      assertThat(client.uploadBlob("abcdef".getBytes(UTF_8))).isEqualTo(digest);
    }
    Mockito.verify(mockByteStreamImpl, Mockito.times(6))
        .write(Mockito.<StreamObserver<WriteResponse>>anyObject());
  }

  @Test
  public void testUploadDirectory() throws Exception {
    final GrpcRemoteCache client = newClient();
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
                .containsExactly(fooDigest, quxDigest, barDigest);
            // Nothing is missing.
            responseObserver.onNext(FindMissingBlobsResponse.getDefaultInstance());
            responseObserver.onCompleted();
          }
        });

    ActionResult.Builder result = ActionResult.newBuilder();
    client.upload(
        execRoot, null, null, null, ImmutableList.<Path>of(fooFile, barDir), outErr, false, result);
    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputFilesBuilder().setPath("a/foo").setDigest(fooDigest);
    expectedResult.addOutputDirectoriesBuilder().setPath("bar").setTreeDigest(barDigest);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void testUploadDirectoryEmpty() throws Exception {
    final GrpcRemoteCache client = newClient();
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
            assertThat(request.getBlobDigestsList()).containsExactly(barDigest);
            // Nothing is missing.
            responseObserver.onNext(FindMissingBlobsResponse.getDefaultInstance());
            responseObserver.onCompleted();
          }
        });

    ActionResult.Builder result = ActionResult.newBuilder();
    client.upload(
        execRoot, null, null, null, ImmutableList.<Path>of(barDir), outErr, false, result);
    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectoriesBuilder().setPath("bar").setTreeDigest(barDigest);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void testUploadDirectoryNested() throws Exception {
    final GrpcRemoteCache client = newClient();
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
                .containsExactly(quxDigest, barDigest, wobbleDigest);
            // Nothing is missing.
            responseObserver.onNext(FindMissingBlobsResponse.getDefaultInstance());
            responseObserver.onCompleted();
          }
        });

    ActionResult.Builder result = ActionResult.newBuilder();
    client.upload(
        execRoot, null, null, null, ImmutableList.<Path>of(barDir), outErr, false, result);
    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectoriesBuilder().setPath("bar").setTreeDigest(barDigest);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void testUploadCacheHits() throws Exception {
    final GrpcRemoteCache client = newClient();
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
    serviceRegistry.addService(
        new ContentAddressableStorageImplBase() {
          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            assertThat(request.getBlobDigestsList())
                .containsExactly(cmdDigest, actionDigest, fooDigest, barDigest);
            // Nothing is missing.
            responseObserver.onNext(FindMissingBlobsResponse.getDefaultInstance());
            responseObserver.onCompleted();
          }
        });

    ActionResult.Builder result = ActionResult.newBuilder();
    client.upload(
        execRoot,
        DIGEST_UTIL.asActionKey(actionDigest),
        action,
        command,
        ImmutableList.<Path>of(fooFile, barFile),
        outErr,
        true,
        result);
    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputFilesBuilder().setPath("a/foo").setDigest(fooDigest);
    expectedResult
        .addOutputFilesBuilder()
        .setPath("bar")
        .setDigest(barDigest)
        .setIsExecutable(true);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void testUploadUploadsOnlyOutputs() throws Exception {
    final GrpcRemoteCache client = newClient();
    final Digest fooDigest =
        fakeFileCache.createScratchInput(ActionInputHelper.fromPath("a/foo"), "xyz");
    final Digest barDigest =
        fakeFileCache.createScratchInput(ActionInputHelper.fromPath("bar"), "x");
    serviceRegistry.addService(
        new ContentAddressableStorageImplBase() {
          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            // This checks we will try to upload the actual outputs.
            assertThat(request.getBlobDigestsList()).containsExactly(fooDigest, barDigest);
            responseObserver.onNext(FindMissingBlobsResponse.getDefaultInstance());
            responseObserver.onCompleted();
          }
        });
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void updateActionResult(
              UpdateActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            fail("Update action result was expected to not be called.");
          }
        });

    ActionKey emptyKey = DIGEST_UTIL.computeActionKey(Action.getDefaultInstance());
    Path fooFile = execRoot.getRelative("a/foo");
    Path barFile = execRoot.getRelative("bar");
    client.upload(
        emptyKey,
        Action.getDefaultInstance(),
        Command.getDefaultInstance(),
        execRoot,
        ImmutableList.<Path>of(fooFile, barFile),
        outErr,
        false);
  }

  @Test
  public void testUploadCacheMissesWithRetries() throws Exception {
    final GrpcRemoteCache client = newClient();
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
    when(mockByteStreamImpl.write(Mockito.<StreamObserver<WriteResponse>>anyObject()))
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
    client.upload(
        actionKey,
        Action.getDefaultInstance(),
        Command.getDefaultInstance(),
        execRoot,
        ImmutableList.<Path>of(fooFile, barFile, bazFile),
        outErr,
        true);
    // 4 times for the errors, 3 times for the successful uploads.
    Mockito.verify(mockByteStreamImpl, Mockito.times(7))
        .write(Mockito.<StreamObserver<WriteResponse>>anyObject());
  }

  @Test
  public void testGetCachedActionResultWithRetries() throws Exception {
    final GrpcRemoteCache client = newClient();
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
    assertThat(client.getCachedActionResult(actionKey)).isNull();
  }
}

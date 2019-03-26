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
import static org.mockito.AdditionalAnswers.answerVoid;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.ActionCacheGrpc.ActionCacheImplBase;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.ContentAddressableStorageGrpc.ContentAddressableStorageImplBase;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.ExecuteRequest;
import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.ExecutionGrpc.ExecutionImplBase;
import build.bazel.remote.execution.v2.FindMissingBlobsRequest;
import build.bazel.remote.execution.v2.FindMissingBlobsResponse;
import build.bazel.remote.execution.v2.GetActionResultRequest;
import build.bazel.remote.execution.v2.OutputFile;
import build.bazel.remote.execution.v2.RequestMetadata;
import build.bazel.remote.execution.v2.WaitExecutionRequest;
import com.google.bytestream.ByteStreamGrpc.ByteStreamImplBase;
import com.google.bytestream.ByteStreamProto.QueryWriteStatusRequest;
import com.google.bytestream.ByteStreamProto.QueryWriteStatusResponse;
import com.google.bytestream.ByteStreamProto.ReadRequest;
import com.google.bytestream.ByteStreamProto.ReadResponse;
import com.google.bytestream.ByteStreamProto.WriteRequest;
import com.google.bytestream.ByteStreamProto.WriteResponse;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.cache.MetadataInjector;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.authandtls.GoogleAuthUtils;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.SpawnExecException;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.exec.util.FakeOwner;
import com.google.devtools.build.lib.remote.RemoteRetrier.ExponentialBackoff;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TestUtils;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.longrunning.Operation;
import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.rpc.Code;
import com.google.rpc.PreconditionFailure;
import com.google.rpc.PreconditionFailure.Violation;
import io.grpc.BindableService;
import io.grpc.CallCredentials;
import io.grpc.Metadata;
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
import java.io.IOException;
import java.time.Duration;
import java.util.Collection;
import java.util.Set;
import java.util.SortedMap;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;
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

/** Tests for {@link RemoteSpawnRunner} in combination with {@link GrpcRemoteExecutor}. */
@RunWith(JUnit4.class)
public class GrpcRemoteExecutionClientTest {

  private static final DigestUtil DIGEST_UTIL = new DigestUtil(DigestHashFunction.SHA256);

  private static final ArtifactExpander SIMPLE_ARTIFACT_EXPANDER =
      new ArtifactExpander() {
        @Override
        public void expand(Artifact artifact, Collection<? super Artifact> output) {
          output.add(artifact);
        }
      };

  private final MutableHandlerRegistry serviceRegistry = new MutableHandlerRegistry();
  private FileSystem fs;
  private Path execRoot;
  private Path logDir;
  private SimpleSpawn simpleSpawn;
  private FakeActionInputFileCache fakeFileCache;
  private Digest inputDigest;
  private Digest cmdDigest;
  private Command command;
  private RemoteSpawnRunner client;
  private FileOutErr outErr;
  private Server fakeServer;
  private static ListeningScheduledExecutorService retryService;

  private static final OutputFile DUMMY_OUTPUT =
      OutputFile.newBuilder().setPath("dummy.txt").build();

  private final SpawnExecutionContext simplePolicy =
      new SpawnExecutionContext() {
        @Override
        public int getId() {
          return 0;
        }

        @Override
        public void prefetchInputs() {
          throw new UnsupportedOperationException();
        }

        @Override
        public void lockOutputFiles() throws InterruptedException {
          throw new UnsupportedOperationException();
        }

        @Override
        public boolean speculating() {
          return false;
        }

        @Override
        public MetadataProvider getMetadataProvider() {
          return fakeFileCache;
        }

        @Override
        public ArtifactExpander getArtifactExpander() {
          throw new UnsupportedOperationException();
        }

        @Override
        public Duration getTimeout() {
          return Duration.ZERO;
        }

        @Override
        public FileOutErr getFileOutErr() {
          return outErr;
        }

        @Override
        public SortedMap<PathFragment, ActionInput> getInputMapping(
            boolean expandTreeArtifactsInRunfiles) throws IOException {
          return new SpawnInputExpander(execRoot, /*strict*/ false)
              .getInputMapping(
                  simpleSpawn,
                  SIMPLE_ARTIFACT_EXPANDER,
                  ArtifactPathResolver.IDENTITY,
                  fakeFileCache,
                  true);
        }

        @Override
        public void report(ProgressStatus state, String name) {
          // TODO(ulfjack): Test that the right calls are made.
        }

        @Override
        public MetadataInjector getMetadataInjector() {
          throw new UnsupportedOperationException();
        }
      };

  @BeforeClass
  public static void beforeEverything() {
    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
  }

  @Before
  public final void setUp() throws Exception {
    String fakeServerName = "fake server for " + getClass();
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
    logDir = fs.getPath("/server-logs");
    FileSystemUtils.createDirectoryAndParents(execRoot);
    fakeFileCache = new FakeActionInputFileCache(execRoot);
    simpleSpawn =
        new SimpleSpawn(
            new FakeOwner("Mnemonic", "Progress Message"),
            ImmutableList.of("/bin/echo", "Hi!"),
            ImmutableMap.of("VARIABLE", "value"),
            /*executionInfo=*/ ImmutableMap.<String, String>of(),
            /*inputs=*/ ImmutableList.of(ActionInputHelper.fromPath("input")),
            /*outputs=*/ ImmutableList.<ActionInput>of(
                new ActionInput() {
                  @Override
                  public String getExecPathString() {
                    return "foo";
                  }

                  @Override
                  public PathFragment getExecPath() {
                    return null; // unused here.
                  }
                },
                new ActionInput() {
                  @Override
                  public String getExecPathString() {
                    return "bar";
                  }

                  @Override
                  public PathFragment getExecPath() {
                    return null; // unused here.
                  }
                }),
            ResourceSet.ZERO);

    Path stdout = fs.getPath("/tmp/stdout");
    Path stderr = fs.getPath("/tmp/stderr");
    FileSystemUtils.createDirectoryAndParents(stdout.getParentDirectory());
    FileSystemUtils.createDirectoryAndParents(stderr.getParentDirectory());
    outErr = new FileOutErr(stdout, stderr);
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new ExponentialBackoff(remoteOptions),
            RemoteRetrier.RETRIABLE_GRPC_EXEC_ERRORS,
            retryService);
    ReferenceCountedChannel channel =
        new ReferenceCountedChannel(InProcessChannelBuilder.forName(fakeServerName).directExecutor().build());
    GrpcRemoteExecutor executor =
        new GrpcRemoteExecutor(channel.retain(), null, retrier);
    CallCredentials creds =
        GoogleAuthUtils.newCallCredentials(Options.getDefaults(AuthAndTLSOptions.class));
    ByteStreamUploader uploader =
        new ByteStreamUploader(remoteOptions.remoteInstanceName, channel.retain(), creds,
            remoteOptions.remoteTimeout, retrier);
    GrpcRemoteCache remoteCache =
        new GrpcRemoteCache(channel.retain(), creds, remoteOptions, retrier, DIGEST_UTIL, uploader);
    client =
        new RemoteSpawnRunner(
            execRoot,
            remoteOptions,
            Options.getDefaults(ExecutionOptions.class),
            null,
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            remoteCache,
            executor,
            RemoteModule.createExecuteRetrier(remoteOptions, retryService),
            DIGEST_UTIL,
            logDir);
    inputDigest = fakeFileCache.createScratchInput(simpleSpawn.getInputFiles().get(0), "xyz");
    command =
        Command.newBuilder()
            .addAllArguments(ImmutableList.of("/bin/echo", "Hi!"))
            .addEnvironmentVariables(
                Command.EnvironmentVariable.newBuilder()
                    .setName("VARIABLE")
                    .setValue("value")
                    .build())
            .addAllOutputFiles(ImmutableList.of("bar", "foo"))
            .build();
    cmdDigest = DIGEST_UTIL.compute(command);
    channel.release();
  }

  @After
  public void tearDown() throws Exception {
    fakeServer.shutdownNow();
    fakeServer.awaitTermination();
  }

  @AfterClass
  public static void afterEverything() {
    retryService.shutdownNow();
  }

  @Test
  public void cacheHit() throws Exception {
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void getActionResult(
              GetActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onNext(ActionResult.getDefaultInstance());
            responseObserver.onCompleted();
          }
        });

    SpawnResult result = client.exec(simpleSpawn, simplePolicy);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.isCacheHit()).isTrue();
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(outErr.hasRecordedOutput()).isFalse();
    assertThat(outErr.hasRecordedStderr()).isFalse();
  }

  @Test
  public void failedAction() throws Exception {
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void getActionResult(
              GetActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onError(Status.NOT_FOUND.asRuntimeException());
          }
        });
    final ActionResult actionResult = ActionResult.newBuilder().setExitCode(1).build();
    serviceRegistry.addService(
        new ExecutionImplBase() {
          @Override
          public void execute(ExecuteRequest request, StreamObserver<Operation> responseObserver) {
            responseObserver.onNext(
                Operation.newBuilder()
                    .setDone(true)
                    .setResponse(
                        Any.pack(ExecuteResponse.newBuilder().setResult(actionResult).build()))
                    .build());
            responseObserver.onCompleted();
          }
        });
    serviceRegistry.addService(
        new ContentAddressableStorageImplBase() {
          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            responseObserver.onNext(FindMissingBlobsResponse.getDefaultInstance());
            responseObserver.onCompleted();
          }
        });

    SpawnResult result = client.exec(simpleSpawn, simplePolicy);
    assertThat(result.exitCode()).isEqualTo(1);
  }

  @Test
  public void noOutputs() throws Exception {
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void getActionResult(
              GetActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onError(Status.NOT_FOUND.asRuntimeException());
          }
        });
    final ActionResult actionResult = ActionResult.getDefaultInstance();
    serviceRegistry.addService(
        new ExecutionImplBase() {
          @Override
          public void execute(ExecuteRequest request, StreamObserver<Operation> responseObserver) {
            responseObserver.onNext(
                Operation.newBuilder()
                    .setDone(true)
                    .setResponse(
                        Any.pack(ExecuteResponse.newBuilder().setResult(actionResult).build()))
                    .build());
            responseObserver.onCompleted();
          }
        });
    serviceRegistry.addService(
        new ContentAddressableStorageImplBase() {
          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            responseObserver.onNext(FindMissingBlobsResponse.getDefaultInstance());
            responseObserver.onCompleted();
          }
        });

    client.exec(simpleSpawn, simplePolicy);
  }

  @Test
  public void cacheHitWithOutput() throws Exception {
    final Digest stdOutDigest = DIGEST_UTIL.computeAsUtf8("stdout");
    final Digest stdErrDigest = DIGEST_UTIL.computeAsUtf8("stderr");
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void getActionResult(
              GetActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onNext(
                ActionResult.newBuilder()
                    .addOutputFiles(DUMMY_OUTPUT)
                    .setStdoutDigest(stdOutDigest)
                    .setStderrDigest(stdErrDigest)
                    .build());
            responseObserver.onCompleted();
          }
        });
    serviceRegistry.addService(
        new FakeImmutableCacheByteStreamImpl(stdOutDigest, "stdout", stdErrDigest, "stderr"));

    SpawnResult result = client.exec(simpleSpawn, simplePolicy);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(result.isCacheHit()).isTrue();
    assertThat(outErr.outAsLatin1()).isEqualTo("stdout");
    assertThat(outErr.errAsLatin1()).isEqualTo("stderr");
  }

  @Test
  public void cacheHitWithInlineOutput() throws Exception {
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void getActionResult(
              GetActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onNext(
                ActionResult.newBuilder()
                    .addOutputFiles(DUMMY_OUTPUT)
                    .setStdoutRaw(ByteString.copyFromUtf8("stdout"))
                    .setStderrRaw(ByteString.copyFromUtf8("stderr"))
                    .build());
            responseObserver.onCompleted();
          }
        });

    SpawnResult result = client.exec(simpleSpawn, simplePolicy);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(result.isCacheHit()).isTrue();
    assertThat(outErr.outAsLatin1()).isEqualTo("stdout");
    assertThat(outErr.errAsLatin1()).isEqualTo("stderr");
  }

  private Answer<StreamObserver<WriteRequest>> blobWriteAnswer(final byte[] data) {
    final Digest digest = DIGEST_UTIL.compute(data);
    return new Answer<StreamObserver<WriteRequest>>() {
      @Override
      public StreamObserver<WriteRequest> answer(InvocationOnMock invocation) {
        @SuppressWarnings("unchecked")
        final StreamObserver<WriteResponse> responseObserver =
            (StreamObserver<WriteResponse>) invocation.getArguments()[0];
        return new StreamObserver<WriteRequest>() {
          @Override
          public void onNext(WriteRequest request) {
            assertThat(request.getResourceName()).contains(DIGEST_UTIL.toString(digest));
            assertThat(request.getFinishWrite()).isTrue();
            assertThat(request.getData().toByteArray()).isEqualTo(data);
            responseObserver.onNext(
                WriteResponse.newBuilder().setCommittedSize(request.getData().size()).build());
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
    };
  }

  private Answer<StreamObserver<WriteRequest>> blobWriteAnswerError() {
    return new Answer<StreamObserver<WriteRequest>>() {
      @Override
      @SuppressWarnings("unchecked")
      public StreamObserver<WriteRequest> answer(final InvocationOnMock invocation) {
        return new StreamObserver<WriteRequest>() {
          @Override
          public void onNext(WriteRequest request) {
            ((StreamObserver<WriteResponse>) invocation.getArguments()[0])
                .onError(Status.UNAVAILABLE.asRuntimeException());
          }

          @Override
          public void onCompleted() {}

          @Override
          public void onError(Throwable t) {
            fail("An unexpected client-side error occurred: " + t);
          }
        };
      }
    };
  }

  /** Capture the request headers from a client. Useful for testing metadata propagation. */
  private static class RequestHeadersValidator implements ServerInterceptor {
    @Override
    public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(
        ServerCall<ReqT, RespT> call,
        Metadata headers,
        ServerCallHandler<ReqT, RespT> next) {
      RequestMetadata meta = headers.get(TracingMetadataUtils.METADATA_KEY);
      assertThat(meta.getCorrelatedInvocationsId()).isEqualTo("build-req-id");
      assertThat(meta.getToolInvocationId()).isEqualTo("command-id");
      assertThat(meta.getActionId()).isNotEmpty();
      assertThat(meta.getToolDetails().getToolName()).isEqualTo("bazel");
      assertThat(meta.getToolDetails().getToolVersion())
          .isEqualTo(BlazeVersionInfo.instance().getVersion());
      return next.startCall(call, headers);
    }
  }

  @Test
  public void remotelyExecute() throws Exception {
    BindableService actionCache =
        new ActionCacheImplBase() {
          @Override
          public void getActionResult(
              GetActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onError(Status.NOT_FOUND.asRuntimeException());
          }
        };
    serviceRegistry.addService(
        ServerInterceptors.intercept(actionCache, new RequestHeadersValidator()));
    final ActionResult actionResult =
        ActionResult.newBuilder()
            .addOutputFiles(DUMMY_OUTPUT)
            .setStdoutRaw(ByteString.copyFromUtf8("stdout"))
            .setStderrRaw(ByteString.copyFromUtf8("stderr"))
            .build();
    BindableService execService =
        new ExecutionImplBase() {
          @Override
          public void execute(ExecuteRequest request, StreamObserver<Operation> responseObserver) {
            responseObserver.onNext(
                Operation.newBuilder()
                    .setDone(true)
                    .setResponse(
                        Any.pack(ExecuteResponse.newBuilder().setResult(actionResult).build()))
                    .build());
            responseObserver.onCompleted();
          }
        };
    serviceRegistry.addService(
        ServerInterceptors.intercept(execService, new RequestHeadersValidator()));
    BindableService cas =
        new ContentAddressableStorageImplBase() {
          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            final Set<Digest> requested = ImmutableSet.copyOf(request.getBlobDigestsList());
            assertThat(requested).contains(cmdDigest);
            assertThat(requested).contains(inputDigest);
            responseObserver.onNext(
                FindMissingBlobsResponse.newBuilder().addMissingBlobDigests(inputDigest).build());
            responseObserver.onCompleted();
          }
        };
    serviceRegistry.addService(ServerInterceptors.intercept(cas, new RequestHeadersValidator()));

    ByteStreamImplBase mockByteStreamImpl = Mockito.mock(ByteStreamImplBase.class);
    when(mockByteStreamImpl.write(Mockito.<StreamObserver<WriteResponse>>anyObject()))
        .thenAnswer(blobWriteAnswer("xyz".getBytes(UTF_8)));
    serviceRegistry.addService(
        ServerInterceptors.intercept(mockByteStreamImpl, new RequestHeadersValidator()));

    SpawnResult result = client.exec(simpleSpawn, simplePolicy);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(result.isCacheHit()).isFalse();
    assertThat(outErr.outAsLatin1()).isEqualTo("stdout");
    assertThat(outErr.errAsLatin1()).isEqualTo("stderr");
    Mockito.verify(mockByteStreamImpl).write(Mockito.<StreamObserver<WriteResponse>>anyObject());
  }

  private Answer<Void> answerWith(@Nullable Operation op, Status status) {
    return invocationOnMock -> {
      @SuppressWarnings("unchecked")
      StreamObserver<Operation> responseObserver =
          (StreamObserver<Operation>) invocationOnMock.getArguments()[1];
      if (op != null) {
        responseObserver.onNext(op);
      }
      if (status.isOk()) {
        responseObserver.onCompleted();
      } else {
        responseObserver.onError(status.asRuntimeException());
      }
      return null;
    };
  }

  @Test
  public void remotelyExecuteRetries() throws Exception {
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
    final Digest resultDigest = DIGEST_UTIL.compute("bla".getBytes(UTF_8));
    final ActionResult actionResult =
        ActionResult.newBuilder()
            .setStdoutRaw(ByteString.copyFromUtf8("stdout"))
            .setStderrRaw(ByteString.copyFromUtf8("stderr"))
            .addOutputFiles(OutputFile.newBuilder().setPath("foo").setDigest(resultDigest).build())
            .build();
    final String opName = "operations/xyz";

    ExecuteResponse executeResponseWithError =
        ExecuteResponse.newBuilder()
            .setStatus(
                com.google.rpc.Status.newBuilder().setCode(Code.INTERNAL.getNumber()).build())
            .build();
    Operation operationWithExecuteError =
        Operation.newBuilder()
            .setName(opName)
            .setDone(true)
            .setResponse(Any.pack(executeResponseWithError))
            .build();
    Operation unfinishedOperation = Operation.newBuilder().setName(opName).build();
    Operation opSuccess =
        Operation.newBuilder()
            .setName(opName)
            .setDone(true)
            .setResponse(Any.pack(ExecuteResponse.newBuilder().setResult(actionResult).build()))
            .build();

    ExecutionImplBase mockExecutionImpl = Mockito.mock(ExecutionImplBase.class);
    // Flow of this test:
    // - call execute, get retriable gRPC error
    // - retry: call execute, get retriable Operation error
    // - retry: call execute, get an Operation, then a retriable gRPC error
    // - retry: call waitExecute, get a retriable gRPC error
    // - retry: call waitExecute, get retriable Operation error
    // - retry: call execute, get successful operation, ignore further errors.
    Mockito.doAnswer(answerWith(null, Status.UNAVAILABLE))
        .doAnswer(answerWith(operationWithExecuteError, Status.OK))
        .doAnswer(answerWith(unfinishedOperation, Status.UNAVAILABLE))
        .doAnswer(answerWith(opSuccess, Status.UNAVAILABLE)) // last status should be ignored.
        .when(mockExecutionImpl)
        .execute(
            Mockito.<ExecuteRequest>anyObject(), Mockito.<StreamObserver<Operation>>anyObject());
    Mockito.doAnswer(answerWith(null, Status.UNAVAILABLE))
        .doAnswer(answerWith(operationWithExecuteError, Status.OK))
        .when(mockExecutionImpl)
        .waitExecution(
            Mockito.<WaitExecutionRequest>anyObject(),
            Mockito.<StreamObserver<Operation>>anyObject());
    serviceRegistry.addService(mockExecutionImpl);

    serviceRegistry.addService(
        new ContentAddressableStorageImplBase() {
          private int numErrors = 4;

          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            if (numErrors-- > 0) {
              responseObserver.onError(Status.UNAVAILABLE.asRuntimeException());
              return;
            }

            final Set<Digest> requested = ImmutableSet.copyOf(request.getBlobDigestsList());
            assertThat(requested).contains(cmdDigest);
            assertThat(requested).contains(inputDigest);
            responseObserver.onNext(
                FindMissingBlobsResponse.newBuilder().addMissingBlobDigests(inputDigest).build());
            responseObserver.onCompleted();
          }
        });

    ByteStreamImplBase mockByteStreamImpl = Mockito.mock(ByteStreamImplBase.class);
    when(mockByteStreamImpl.write(Mockito.<StreamObserver<WriteResponse>>anyObject()))
        .thenAnswer(blobWriteAnswerError()) // Error on the input file.
        .thenAnswer(blobWriteAnswerError()) // Error on the input file again.
        .thenAnswer(blobWriteAnswer("xyz".getBytes(UTF_8))); // Upload input file successfully.
    doAnswer(
        answerVoid((QueryWriteStatusRequest request, StreamObserver<QueryWriteStatusResponse> responseObserver) -> {
            responseObserver.onNext(QueryWriteStatusResponse.newBuilder()
                .setCommittedSize(0)
                .setComplete(false)
                .build());
            responseObserver.onCompleted();
        }))
        .when(mockByteStreamImpl).queryWriteStatus(any(QueryWriteStatusRequest.class), any(StreamObserver.class));
    Mockito.doAnswer(
            invocationOnMock -> {
              @SuppressWarnings("unchecked")
              StreamObserver<ReadResponse> responseObserver =
                  (StreamObserver<ReadResponse>) invocationOnMock.getArguments()[1];
              responseObserver.onError(Status.INTERNAL.asRuntimeException()); // Will retry.
              return null;
            })
        .doAnswer(
            invocationOnMock -> {
              @SuppressWarnings("unchecked")
              StreamObserver<ReadResponse> responseObserver =
                  (StreamObserver<ReadResponse>) invocationOnMock.getArguments()[1];
              responseObserver.onNext(
                  ReadResponse.newBuilder().setData(ByteString.copyFromUtf8("bla")).build());
              responseObserver.onCompleted();
              return null;
            })
        .when(mockByteStreamImpl)
        .read(Mockito.<ReadRequest>anyObject(), Mockito.<StreamObserver<ReadResponse>>anyObject());
    serviceRegistry.addService(mockByteStreamImpl);

    SpawnResult result = client.exec(simpleSpawn, simplePolicy);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(result.isCacheHit()).isFalse();
    assertThat(outErr.outAsLatin1()).isEqualTo("stdout");
    assertThat(outErr.errAsLatin1()).isEqualTo("stderr");
    Mockito.verify(mockExecutionImpl, Mockito.times(4))
        .execute(
            Mockito.<ExecuteRequest>anyObject(), Mockito.<StreamObserver<Operation>>anyObject());
    Mockito.verify(mockExecutionImpl, Mockito.times(2))
        .waitExecution(
            Mockito.<WaitExecutionRequest>anyObject(),
            Mockito.<StreamObserver<Operation>>anyObject());
    Mockito.verify(mockByteStreamImpl, Mockito.times(2))
        .read(Mockito.<ReadRequest>anyObject(), Mockito.<StreamObserver<ReadResponse>>anyObject());
    Mockito.verify(mockByteStreamImpl, Mockito.times(3))
        .write(Mockito.<StreamObserver<WriteResponse>>anyObject());
  }

  @Test
  public void remotelyExecuteRetriesWaitResult() throws Exception {
    // This test's flow is similar to the previous, except the result
    // will eventually be returned by the waitExecute function.
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void getActionResult(
              GetActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onError(Status.NOT_FOUND.asRuntimeException());
          }
        });
    final Digest resultDigest = DIGEST_UTIL.compute("bla".getBytes(UTF_8));
    final ActionResult actionResult =
        ActionResult.newBuilder()
            .setStdoutRaw(ByteString.copyFromUtf8("stdout"))
            .setStderrRaw(ByteString.copyFromUtf8("stderr"))
            .addOutputFiles(OutputFile.newBuilder().setPath("foo").setDigest(resultDigest).build())
            .build();
    final String opName = "operations/xyz";

    Operation unfinishedOperation = Operation.newBuilder().setName(opName).build();
    Operation opSuccess =
        Operation.newBuilder()
            .setName(opName)
            .setDone(true)
            .setResponse(Any.pack(ExecuteResponse.newBuilder().setResult(actionResult).build()))
            .build();

    ExecutionImplBase mockExecutionImpl = Mockito.mock(ExecutionImplBase.class);
    // Flow of this test:
    // - call execute, get an Operation, then a retriable gRPC error
    // - retry: call waitExecute, get NOT_FOUND (operation lost)
    // - retry: call execute, get NOT_FOUND (operation lost)
    // - retry: call execute, get an Operation, then a retriable gRPC error
    // - retry: call waitExecute, get successful operation, ignore further errors.
    Mockito.doAnswer(answerWith(unfinishedOperation, Status.UNAVAILABLE))
        .doAnswer(answerWith(unfinishedOperation, Status.NOT_FOUND))
        .doAnswer(answerWith(unfinishedOperation, Status.UNAVAILABLE))
        .when(mockExecutionImpl)
        .execute(
            Mockito.<ExecuteRequest>anyObject(), Mockito.<StreamObserver<Operation>>anyObject());
    Mockito.doAnswer(answerWith(unfinishedOperation, Status.NOT_FOUND))
        .doAnswer(answerWith(opSuccess, Status.UNAVAILABLE)) // This error is ignored.
        .when(mockExecutionImpl)
        .waitExecution(
            Mockito.<WaitExecutionRequest>anyObject(),
            Mockito.<StreamObserver<Operation>>anyObject());
    serviceRegistry.addService(mockExecutionImpl);

    serviceRegistry.addService(
        new ContentAddressableStorageImplBase() {

          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            final Set<Digest> requested = ImmutableSet.copyOf(request.getBlobDigestsList());
            assertThat(requested).contains(cmdDigest);
            assertThat(requested).contains(inputDigest);
            responseObserver.onNext(
                FindMissingBlobsResponse.newBuilder().addMissingBlobDigests(inputDigest).build());
            responseObserver.onCompleted();
          }
        });

    ByteStreamImplBase mockByteStreamImpl = Mockito.mock(ByteStreamImplBase.class);
    when(mockByteStreamImpl.write(Mockito.<StreamObserver<WriteResponse>>anyObject()))
        .thenAnswer(blobWriteAnswer("xyz".getBytes(UTF_8))); // Upload input file successfully.
    Mockito.doAnswer(
            invocationOnMock -> {
              @SuppressWarnings("unchecked")
              StreamObserver<ReadResponse> responseObserver =
                  (StreamObserver<ReadResponse>) invocationOnMock.getArguments()[1];
              responseObserver.onNext(
                  ReadResponse.newBuilder().setData(ByteString.copyFromUtf8("bla")).build());
              responseObserver.onCompleted();
              return null;
            })
        .when(mockByteStreamImpl)
        .read(Mockito.<ReadRequest>anyObject(), Mockito.<StreamObserver<ReadResponse>>anyObject());
    serviceRegistry.addService(mockByteStreamImpl);

    SpawnResult result = client.exec(simpleSpawn, simplePolicy);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(result.isCacheHit()).isFalse();
    assertThat(outErr.outAsLatin1()).isEqualTo("stdout");
    assertThat(outErr.errAsLatin1()).isEqualTo("stderr");
    Mockito.verify(mockExecutionImpl, Mockito.times(3))
        .execute(
            Mockito.<ExecuteRequest>anyObject(), Mockito.<StreamObserver<Operation>>anyObject());
    Mockito.verify(mockExecutionImpl, Mockito.times(2))
        .waitExecution(
            Mockito.<WaitExecutionRequest>anyObject(),
            Mockito.<StreamObserver<Operation>>anyObject());
    Mockito.verify(mockByteStreamImpl)
        .read(Mockito.<ReadRequest>anyObject(), Mockito.<StreamObserver<ReadResponse>>anyObject());
    Mockito.verify(mockByteStreamImpl, Mockito.times(1))
        .write(Mockito.<StreamObserver<WriteResponse>>anyObject());
  }

  @Test
  public void passUnavailableErrorWithStackTrace() throws Exception {
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void getActionResult(
              GetActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onError(Status.UNAVAILABLE.asRuntimeException());
          }
        });

    try {
      client.exec(simpleSpawn, simplePolicy);
      fail("Expected an exception");
    } catch (SpawnExecException expected) {
      assertThat(expected.getSpawnResult().status())
          .isEqualTo(SpawnResult.Status.EXECUTION_FAILED_CATASTROPHICALLY);
      // Ensure we also got back the stack trace.
      assertThat(expected).hasMessageThat()
          .contains("GrpcRemoteExecutionClientTest.passUnavailableErrorWithStackTrace");
    }
  }

  @Test
  public void passInternalErrorWithStackTrace() throws Exception {
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void getActionResult(
              GetActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onError(Status.INTERNAL.withDescription("whoa").asRuntimeException());
          }
        });

    try {
      client.exec(simpleSpawn, simplePolicy);
      fail("Expected an exception");
    } catch (ExecException expected) {
      assertThat(expected).hasMessageThat().contains("whoa"); // Error details.
      // Ensure we also got back the stack trace.
      assertThat(expected).hasMessageThat()
          .contains("GrpcRemoteExecutionClientTest.passInternalErrorWithStackTrace");
    }
  }

  @Test
  public void passCacheMissErrorWithStackTrace() throws Exception {
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void getActionResult(
              GetActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onError(Status.NOT_FOUND.asRuntimeException());
          }
        });
    Digest stdOutDigest = DIGEST_UTIL.computeAsUtf8("bla");
    final ActionResult actionResult =
        ActionResult.newBuilder()
            .addOutputFiles(DUMMY_OUTPUT)
            .setStdoutDigest(stdOutDigest)
            .build();
    serviceRegistry.addService(
        new ExecutionImplBase() {
          @Override
          public void execute(ExecuteRequest request, StreamObserver<Operation> responseObserver) {
            responseObserver.onNext(
                Operation.newBuilder()
                    .setDone(true)
                    .setResponse(
                        Any.pack(ExecuteResponse.newBuilder().setResult(actionResult).build()))
                    .build());
            responseObserver.onCompleted();
          }
        });
    serviceRegistry.addService(
        new ContentAddressableStorageImplBase() {
          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            responseObserver.onNext(FindMissingBlobsResponse.getDefaultInstance());
            responseObserver.onCompleted();
          }
        });
    serviceRegistry.addService(
        new ByteStreamImplBase() {
          @Override
          public void read(ReadRequest request, StreamObserver<ReadResponse> responseObserver) {
            assertThat(request.getResourceName().contains(DIGEST_UTIL.toString(stdOutDigest)))
                .isTrue();
            responseObserver.onError(Status.NOT_FOUND.asRuntimeException());
          }
        });

    try {
      client.exec(simpleSpawn, simplePolicy);
      fail("Expected an exception");
    } catch (SpawnExecException expected) {
      assertThat(expected.getSpawnResult().status())
          .isEqualTo(SpawnResult.Status.REMOTE_CACHE_FAILED);
      assertThat(expected).hasMessageThat().contains(DIGEST_UTIL.toString(stdOutDigest));
      // Ensure we also got back the stack trace.
      assertThat(expected).hasMessageThat()
          .contains("GrpcRemoteExecutionClientTest.passCacheMissErrorWithStackTrace");
    }
  }

  @Test
  public void passRepeatedOrphanedCacheMissErrorWithStackTrace() throws Exception {
    final Digest stdOutDigest = DIGEST_UTIL.computeAsUtf8("bloo");
    final ActionResult actionResult =
        ActionResult.newBuilder()
            .addOutputFiles(DUMMY_OUTPUT)
            .setStdoutDigest(stdOutDigest)
            .build();
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void getActionResult(
              GetActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onNext(actionResult);
            responseObserver.onCompleted();
          }
        });
    serviceRegistry.addService(
        new ExecutionImplBase() {
          @Override
          public void execute(ExecuteRequest request, StreamObserver<Operation> responseObserver) {
            responseObserver.onNext(
                Operation.newBuilder()
                    .setDone(true)
                    .setResponse(
                        Any.pack(ExecuteResponse.newBuilder().setResult(actionResult).build()))
                    .build());
            responseObserver.onCompleted();
          }
        });
    serviceRegistry.addService(
        new ContentAddressableStorageImplBase() {
          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            responseObserver.onNext(FindMissingBlobsResponse.getDefaultInstance());
            responseObserver.onCompleted();
          }
        });
    serviceRegistry.addService(
        new ByteStreamImplBase() {
          @Override
          public void read(ReadRequest request, StreamObserver<ReadResponse> responseObserver) {
            assertThat(request.getResourceName().contains(DIGEST_UTIL.toString(stdOutDigest)))
                .isTrue();
            responseObserver.onError(Status.NOT_FOUND.asRuntimeException());
          }
        });

    try {
      client.exec(simpleSpawn, simplePolicy);
      fail("Expected an exception");
    } catch (SpawnExecException expected) {
      assertThat(expected.getSpawnResult().status())
          .isEqualTo(SpawnResult.Status.REMOTE_CACHE_FAILED);
      assertThat(expected).hasMessageThat().contains(DIGEST_UTIL.toString(stdOutDigest));
      // Ensure we also got back the stack trace.
      assertThat(expected)
          .hasMessageThat()
          .contains("passRepeatedOrphanedCacheMissErrorWithStackTrace");
    }
  }

  @Test
  public void remotelyReExecuteOrphanedCachedActions() throws Exception {
    final Digest stdOutDigest = DIGEST_UTIL.computeAsUtf8("stdout");
    final ActionResult actionResult =
        ActionResult.newBuilder()
            .addOutputFiles(DUMMY_OUTPUT)
            .setStdoutDigest(stdOutDigest)
            .build();
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void getActionResult(
              GetActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onNext(actionResult);
            responseObserver.onCompleted();
          }
        });
    serviceRegistry.addService(
        new ByteStreamImplBase() {
          private boolean first = true;

          @Override
          public void read(ReadRequest request, StreamObserver<ReadResponse> responseObserver) {
            // First read is a cache miss, next read succeeds.
            if (first) {
              first = false;
              responseObserver.onError(Status.NOT_FOUND.asRuntimeException());
            } else {
              responseObserver.onNext(
                  ReadResponse.newBuilder().setData(ByteString.copyFromUtf8("stdout")).build());
              responseObserver.onCompleted();
            }
          }

          @Override
          public StreamObserver<WriteRequest> write(
              StreamObserver<WriteResponse> responseObserver) {
            return new StreamObserver<WriteRequest>() {
              @Override
              public void onNext(WriteRequest request) {}

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
    AtomicInteger numExecuteCalls = new AtomicInteger();
    serviceRegistry.addService(
        new ExecutionImplBase() {
          @Override
          public void execute(ExecuteRequest request, StreamObserver<Operation> responseObserver) {
            numExecuteCalls.incrementAndGet();
            assertThat(request.getSkipCacheLookup()).isTrue(); // Action will be re-executed.
            responseObserver.onNext(
                Operation.newBuilder()
                    .setDone(true)
                    .setResponse(
                        Any.pack(ExecuteResponse.newBuilder().setResult(actionResult).build()))
                    .build());
            responseObserver.onCompleted();
          }
        });
    serviceRegistry.addService(
        new ContentAddressableStorageImplBase() {
          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            // Nothing is missing.
            responseObserver.onNext(FindMissingBlobsResponse.getDefaultInstance());
            responseObserver.onCompleted();
          }
        });

    SpawnResult result = client.exec(simpleSpawn, simplePolicy);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(result.isCacheHit()).isFalse();
    assertThat(outErr.outAsLatin1()).isEqualTo("stdout");
    assertThat(numExecuteCalls.get()).isEqualTo(1);
  }

  @Test
  public void retryUploadAndExecuteOnMissingInputs() throws Exception {
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void getActionResult(
              GetActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onError(Status.NOT_FOUND.asRuntimeException());
          }
        });
    serviceRegistry.addService(
        new ByteStreamImplBase() {
          @Override
          public void read(ReadRequest request, StreamObserver<ReadResponse> responseObserver) {
            responseObserver.onNext(
                ReadResponse.newBuilder().setData(ByteString.copyFromUtf8("bla")).build());
            responseObserver.onCompleted();
          }

          @Override
          public StreamObserver<WriteRequest> write(
              StreamObserver<WriteResponse> responseObserver) {
            return new StreamObserver<WriteRequest>() {
              @Override
              public void onNext(WriteRequest request) {}

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
    final ActionResult actionResult =
        ActionResult.newBuilder()
            .addOutputFiles(DUMMY_OUTPUT)
            .build();
    AtomicInteger numExecuteCalls = new AtomicInteger();
    serviceRegistry.addService(
        new ExecutionImplBase() {
          @Override
          public void execute(ExecuteRequest request, StreamObserver<Operation> responseObserver) {
            if (numExecuteCalls.incrementAndGet() == 1) {
              // Missing input.
              Violation viol = Violation.newBuilder().setType("MISSING").build();
              com.google.rpc.Status status =
                  com.google.rpc.Status.newBuilder()
                      .setCode(Code.FAILED_PRECONDITION.getNumber())
                      .addDetails(
                          Any.pack(PreconditionFailure.newBuilder().addViolations(viol).build()))
                      .build();
              responseObserver.onNext(
                  Operation.newBuilder()
                      .setDone(true)
                      .setResponse(Any.pack(ExecuteResponse.newBuilder().setStatus(status).build()))
                      .build());
              responseObserver.onCompleted();
            } else {
              assertThat(request.getSkipCacheLookup()).isFalse();
              responseObserver.onNext(
                  Operation.newBuilder()
                      .setDone(true)
                      .setResponse(
                          Any.pack(ExecuteResponse.newBuilder().setResult(actionResult).build()))
                      .build());
              responseObserver.onCompleted();
            }
          }
        });
    AtomicInteger numCacheUploads = new AtomicInteger();
    serviceRegistry.addService(
        new ContentAddressableStorageImplBase() {
          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            numCacheUploads.incrementAndGet();
            // Nothing is missing.
            responseObserver.onNext(FindMissingBlobsResponse.getDefaultInstance());
            responseObserver.onCompleted();
          }
        });

    SpawnResult result = client.exec(simpleSpawn, simplePolicy);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(result.isCacheHit()).isFalse();
    assertThat(numCacheUploads.get()).isEqualTo(2);
    assertThat(numExecuteCalls.get()).isEqualTo(2);
  }
}

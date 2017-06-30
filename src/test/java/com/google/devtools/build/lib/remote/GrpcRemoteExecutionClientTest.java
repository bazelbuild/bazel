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
import static org.mockito.Mockito.when;

import com.google.bytestream.ByteStreamGrpc.ByteStreamImplBase;
import com.google.bytestream.ByteStreamProto.WriteRequest;
import com.google.bytestream.ByteStreamProto.WriteResponse;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.exec.SpawnResult;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionPolicy;
import com.google.devtools.build.lib.exec.util.FakeOwner;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystem.HashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.devtools.remoteexecution.v1test.ActionCacheGrpc.ActionCacheImplBase;
import com.google.devtools.remoteexecution.v1test.ActionResult;
import com.google.devtools.remoteexecution.v1test.Command;
import com.google.devtools.remoteexecution.v1test.ContentAddressableStorageGrpc.ContentAddressableStorageImplBase;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.devtools.remoteexecution.v1test.ExecuteRequest;
import com.google.devtools.remoteexecution.v1test.ExecuteResponse;
import com.google.devtools.remoteexecution.v1test.ExecutionGrpc.ExecutionImplBase;
import com.google.devtools.remoteexecution.v1test.FindMissingBlobsRequest;
import com.google.devtools.remoteexecution.v1test.FindMissingBlobsResponse;
import com.google.devtools.remoteexecution.v1test.GetActionResultRequest;
import com.google.longrunning.Operation;
import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.watcher.v1.Change;
import com.google.watcher.v1.ChangeBatch;
import com.google.watcher.v1.Request;
import com.google.watcher.v1.WatcherGrpc.WatcherImplBase;
import io.grpc.Channel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import io.grpc.util.MutableHandlerRegistry;
import java.io.IOException;
import java.util.Collection;
import java.util.Set;
import java.util.SortedMap;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;
import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;

/** Tests for {@link RemoteSpawnRunner} in combination with {@link GrpcRemoteExecutor}. */
@RunWith(JUnit4.class)
public class GrpcRemoteExecutionClientTest {
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
  private SimpleSpawn simpleSpawn;
  private FakeActionInputFileCache fakeFileCache;
  private Digest inputDigest;
  private RemoteSpawnRunner client;
  private FileOutErr outErr;
  private Server fakeServer;

  private final SpawnExecutionPolicy simplePolicy =
      new SpawnExecutionPolicy() {
        @Override
        public void lockOutputFiles() throws InterruptedException {
          throw new UnsupportedOperationException();
        }

        @Override
        public ActionInputFileCache getActionInputFileCache() {
          return fakeFileCache;
        }

        @Override
        public long getTimeoutMillis() {
          return 0;
        }

        @Override
        public FileOutErr getFileOutErr() {
          return outErr;
        }

        @Override
        public SortedMap<PathFragment, ActionInput> getInputMapping() throws IOException {
          return new SpawnInputExpander(/*strict*/ false)
              .getInputMapping(simpleSpawn, SIMPLE_ARTIFACT_EXPANDER, fakeFileCache, "workspace");
        }

        @Override
        public void report(ProgressStatus state) {
          // TODO(ulfjack): Test that the right calls are made.
        }
      };

  @Before
  public final void setUp() throws Exception {
    FileSystem.setDigestFunctionForTesting(HashFunction.SHA1);
    String fakeServerName = "fake server for " + getClass();
    // Use a mutable service registry for later registering the service impl for each test case.
    fakeServer =
        InProcessServerBuilder.forName(fakeServerName)
            .fallbackHandlerRegistry(serviceRegistry)
            .directExecutor()
            .build()
            .start();

    Chunker.setDefaultChunkSizeForTesting(1000); // Enough for everything to be one chunk.
    fs = new InMemoryFileSystem();
    execRoot = fs.getPath("/exec/root");
    FileSystemUtils.createDirectoryAndParents(execRoot);
    fakeFileCache = new FakeActionInputFileCache(execRoot);
    simpleSpawn =
        new SimpleSpawn(
            new FakeOwner("Mnemonic", "Progress Message"),
            ImmutableList.of("/bin/echo", "Hi!"),
            ImmutableMap.of("VARIABLE", "value"),
            /*executionInfo=*/ ImmutableMap.<String, String>of(),
            /*inputs=*/ ImmutableList.of(ActionInputHelper.fromPath("input")),
            /*outputs=*/ ImmutableList.<ActionInput>of(),
            ResourceSet.ZERO);

    Path stdout = fs.getPath("/tmp/stdout");
    Path stderr = fs.getPath("/tmp/stderr");
    FileSystemUtils.createDirectoryAndParents(stdout.getParentDirectory());
    FileSystemUtils.createDirectoryAndParents(stderr.getParentDirectory());
    outErr = new FileOutErr(stdout, stderr);
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    Channel channel = InProcessChannelBuilder.forName(fakeServerName).directExecutor().build();
    GrpcRemoteExecutor executor = new GrpcRemoteExecutor(channel, ChannelOptions.DEFAULT, options);
    GrpcActionCache remoteCache = new GrpcActionCache(channel, ChannelOptions.DEFAULT, options);
    client = new RemoteSpawnRunner(execRoot, options, executor, remoteCache);
    inputDigest = fakeFileCache.createScratchInput(simpleSpawn.getInputFiles().get(0), "xyz");
  }

  @After
  public void tearDown() throws Exception {
    fakeServer.shutdownNow();
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
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(outErr.hasRecordedOutput()).isFalse();
    assertThat(outErr.hasRecordedStderr()).isFalse();
  }

  @Test
  public void cacheHitWithOutput() throws Exception {
    final Digest stdOutDigest = Digests.computeDigestUtf8("stdout");
    final Digest stdErrDigest = Digests.computeDigestUtf8("stderr");
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void getActionResult(
              GetActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onNext(
                ActionResult.newBuilder()
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
                    .setStdoutRaw(ByteString.copyFromUtf8("stdout"))
                    .setStderrRaw(ByteString.copyFromUtf8("stderr"))
                    .build());
            responseObserver.onCompleted();
          }
        });

    SpawnResult result = client.exec(simpleSpawn, simplePolicy);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(outErr.outAsLatin1()).isEqualTo("stdout");
    assertThat(outErr.errAsLatin1()).isEqualTo("stderr");
  }

  private Answer<StreamObserver<WriteRequest>> blobWriteAnswer(final byte[] data) {
    final Digest digest = Digests.computeDigest(data);
    return new Answer<StreamObserver<WriteRequest>>() {
      @Override
      public StreamObserver<WriteRequest> answer(InvocationOnMock invocation) {
        @SuppressWarnings("unchecked")
        final StreamObserver<WriteResponse> responseObserver =
            (StreamObserver<WriteResponse>) invocation.getArguments()[0];
        return new StreamObserver<WriteRequest>() {
          @Override
          public void onNext(WriteRequest request) {
            assertThat(request.getResourceName()).contains(digest.getHash());
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

  @Test
  public void remotelyExecute() throws Exception {
    serviceRegistry.addService(
        new ActionCacheImplBase() {
          @Override
          public void getActionResult(
              GetActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
            responseObserver.onError(Status.NOT_FOUND.asRuntimeException());
          }
        });
    final ActionResult actionResult =
        ActionResult.newBuilder()
            .setStdoutRaw(ByteString.copyFromUtf8("stdout"))
            .setStderrRaw(ByteString.copyFromUtf8("stderr"))
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
    final Command command =
        Command.newBuilder()
            .addAllArguments(ImmutableList.of("/bin/echo", "Hi!"))
            .addEnvironmentVariables(
                Command.EnvironmentVariable.newBuilder()
                    .setName("VARIABLE")
                    .setValue("value")
                    .build())
            .build();
    final Digest cmdDigest = Digests.computeDigest(command);
    serviceRegistry.addService(
        new ContentAddressableStorageImplBase() {
          @Override
          public void findMissingBlobs(
              FindMissingBlobsRequest request,
              StreamObserver<FindMissingBlobsResponse> responseObserver) {
            FindMissingBlobsResponse.Builder b = FindMissingBlobsResponse.newBuilder();
            final Set<Digest> requested = ImmutableSet.copyOf(request.getBlobDigestsList());
            if (requested.contains(cmdDigest)) {
              b.addMissingBlobDigests(cmdDigest);
            } else if (requested.contains(inputDigest)) {
              b.addMissingBlobDigests(inputDigest);
            } else {
              fail("Unexpected call to findMissingBlobs: " + request);
            }
            responseObserver.onNext(b.build());
            responseObserver.onCompleted();
          }
        });

    ByteStreamImplBase mockByteStreamImpl = Mockito.mock(ByteStreamImplBase.class);
    when(mockByteStreamImpl.write(Mockito.<StreamObserver<WriteResponse>>anyObject()))
        .thenAnswer(blobWriteAnswer(command.toByteArray()))
        .thenAnswer(blobWriteAnswer("xyz".getBytes(UTF_8)));
    serviceRegistry.addService(mockByteStreamImpl);

    SpawnResult result = client.exec(simpleSpawn, simplePolicy);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(outErr.outAsLatin1()).isEqualTo("stdout");
    assertThat(outErr.errAsLatin1()).isEqualTo("stderr");
  }

  @Test
  public void remotelyExecuteWithWatchAndRetries() throws Exception {
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
    final ActionResult actionResult =
        ActionResult.newBuilder()
            .setStdoutRaw(ByteString.copyFromUtf8("stdout"))
            .setStderrRaw(ByteString.copyFromUtf8("stderr"))
            .build();
    final String opName = "operations/xyz";
    serviceRegistry.addService(
        new ExecutionImplBase() {
          private int numErrors = 4;

          @Override
          public void execute(ExecuteRequest request, StreamObserver<Operation> responseObserver) {
            if (numErrors-- <= 0) {
              responseObserver.onNext(Operation.newBuilder().setName(opName).build());
              responseObserver.onCompleted();
            } else {
              responseObserver.onError(Status.UNAVAILABLE.asRuntimeException());
            }
          }
        });
    serviceRegistry.addService(
        new WatcherImplBase() {
          private int numErrors = 4;

          @Override
          public void watch(Request request, StreamObserver<ChangeBatch> responseObserver) {
            assertThat(request.getTarget()).isEqualTo(opName);
            if (numErrors-- > 0) {
              responseObserver.onError(Status.UNAVAILABLE.asRuntimeException());
              return;
            }
            // Some optional initial state.
            responseObserver.onNext(
                ChangeBatch.newBuilder()
                    .addChanges(
                        Change.newBuilder().setState(Change.State.INITIAL_STATE_SKIPPED).build())
                    .build());
            // Still executing.
            responseObserver.onNext(
                ChangeBatch.newBuilder()
                    .addChanges(
                        Change.newBuilder()
                            .setState(Change.State.EXISTS)
                            .setData(Any.pack(Operation.newBuilder().setName(opName).build()))
                            .build())
                    .addChanges(
                        Change.newBuilder()
                            .setState(Change.State.EXISTS)
                            .setData(Any.pack(Operation.newBuilder().setName(opName).build()))
                            .build())
                    .build());
            // Finished executing.
            responseObserver.onNext(
                ChangeBatch.newBuilder()
                    .addChanges(
                        Change.newBuilder()
                            .setState(Change.State.EXISTS)
                            .setData(
                                Any.pack(
                                    Operation.newBuilder()
                                        .setName(opName)
                                        .setDone(true)
                                        .setResponse(
                                            Any.pack(
                                                ExecuteResponse.newBuilder()
                                                    .setResult(actionResult)
                                                    .build()))
                                        .build()))
                            .build())
                    .build());
            responseObserver.onCompleted();
          }
        });
    final Command command =
        Command.newBuilder()
            .addAllArguments(ImmutableList.of("/bin/echo", "Hi!"))
            .addEnvironmentVariables(
                Command.EnvironmentVariable.newBuilder()
                    .setName("VARIABLE")
                    .setValue("value")
                    .build())
            .build();
    final Digest cmdDigest = Digests.computeDigest(command);
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

            FindMissingBlobsResponse.Builder b = FindMissingBlobsResponse.newBuilder();
            final Set<Digest> requested = ImmutableSet.copyOf(request.getBlobDigestsList());
            if (requested.contains(cmdDigest)) {
              b.addMissingBlobDigests(cmdDigest);
            } else if (requested.contains(inputDigest)) {
              b.addMissingBlobDigests(inputDigest);
            } else {
              fail("Unexpected call to findMissingBlobs: " + request);
            }
            responseObserver.onNext(b.build());
            responseObserver.onCompleted();
          }
        });

    ByteStreamImplBase mockByteStreamImpl = Mockito.mock(ByteStreamImplBase.class);
    when(mockByteStreamImpl.write(Mockito.<StreamObserver<WriteResponse>>anyObject()))
        .thenAnswer(blobWriteAnswerError())                  // Error on command upload.
        .thenAnswer(blobWriteAnswer(command.toByteArray()))  // Upload command successfully.
        .thenAnswer(blobWriteAnswerError())                  // Error on the input file.
        .thenAnswer(blobWriteAnswerError())                  // Error on the input file again.
        .thenAnswer(blobWriteAnswer("xyz".getBytes(UTF_8))); // Upload input file successfully.
    serviceRegistry.addService(mockByteStreamImpl);

    SpawnResult result = client.exec(simpleSpawn, simplePolicy);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(outErr.outAsLatin1()).isEqualTo("stdout");
    assertThat(outErr.errAsLatin1()).isEqualTo("stderr");
  }
}

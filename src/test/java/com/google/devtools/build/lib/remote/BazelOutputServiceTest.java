// Copyright 2026 The Bazel Authors. All rights reserved.
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
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.actions.cache.OutputMetadataStore;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.remote.BazelOutputServiceProto.FinalizeArtifactsRequest;
import com.google.devtools.build.lib.remote.BazelOutputServiceProto.FinalizeArtifactsResponse;
import com.google.devtools.build.lib.remote.BazelOutputServiceProto.StageArtifactsRequest;
import com.google.devtools.build.lib.remote.BazelOutputServiceProto.StageArtifactsResponse;
import com.google.devtools.build.lib.remote.BazelOutputServiceProto.StartBuildRequest;
import com.google.devtools.build.lib.remote.BazelOutputServiceProto.StartBuildResponse;
import com.google.devtools.build.lib.remote.RemoteExecutionService.ActionResultMetadata.FileMetadata;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.rpc.Status;
import io.grpc.Channel;
import io.grpc.Server;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class BazelOutputServiceTest {

  private static final String SERVER_NAME = "fake-server";

  private static class FakeBazelOutputService
      extends BazelOutputServiceGrpc.BazelOutputServiceImplBase {
    private final List<StageArtifactsRequest> stageRequests = new ArrayList<>();
    private final List<FinalizeArtifactsRequest> finalizeRequests = new ArrayList<>();

    @Override
    public void startBuild(
        StartBuildRequest request, StreamObserver<StartBuildResponse> responseObserver) {
      responseObserver.onNext(
          StartBuildResponse.newBuilder().setOutputPathSuffix("suffix").build());
      responseObserver.onCompleted();
    }

    @Override
    public void stageArtifacts(
        StageArtifactsRequest request, StreamObserver<StageArtifactsResponse> responseObserver) {
      stageRequests.add(request);
      StageArtifactsResponse.Builder response = StageArtifactsResponse.newBuilder();
      for (int i = 0; i < request.getArtifactsCount(); i++) {
        response.addResponses(
            StageArtifactsResponse.Response.newBuilder()
                .setStatus(Status.newBuilder().setCode(0).build()) // OK
                .build());
      }
      responseObserver.onNext(response.build());
      responseObserver.onCompleted();
    }

    @Override
    public void finalizeArtifacts(
        FinalizeArtifactsRequest request,
        StreamObserver<FinalizeArtifactsResponse> responseObserver) {
      finalizeRequests.add(request);
      responseObserver.onNext(FinalizeArtifactsResponse.getDefaultInstance());
      responseObserver.onCompleted();
    }

    List<StageArtifactsRequest> getStageRequests() {
      return stageRequests;
    }

    List<FinalizeArtifactsRequest> getFinalizeRequests() {
      return finalizeRequests;
    }
  }

  private final FakeBazelOutputService fakeService = new FakeBazelOutputService();
  private Server server;
  private Channel inProcessChannel;
  private FileSystem fileSystem;
  private Path outputBase;
  private Path execRoot;
  private Path outputPath;
  private DigestUtil digestUtil;
  private ArtifactRoot artifactRoot;

  @Before
  public void setUp() throws Exception {
    server =
        InProcessServerBuilder.forName(SERVER_NAME)
            .addService(fakeService)
            .directExecutor()
            .build()
            .start();

    inProcessChannel = InProcessChannelBuilder.forName(SERVER_NAME).directExecutor().build();
    fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);
    outputBase = fileSystem.getPath("/output_base");
    execRoot = fileSystem.getPath("/workspace");
    outputPath = execRoot.getRelative("outputs");
    digestUtil = new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
    artifactRoot = ArtifactRoot.asDerivedRoot(execRoot, ArtifactRoot.RootType.OUTPUT, "outputs");
    artifactRoot.getRoot().asPath().createDirectoryAndParents();
  }

  @After
  public void tearDown() throws Exception {
    inProcessChannel = null;
    server.shutdownNow();
    server.awaitTermination();
  }

  private BazelOutputService createService(int maxOutboundMessageSize) throws Exception {
    ReferenceCountedChannel channel = mock(ReferenceCountedChannel.class);
    when(channel.withChannelBlocking(any()))
        .thenAnswer(
            invocation -> {
              ReferenceCountedChannel.IOFunction<Channel, ?> formula = invocation.getArgument(0);
              return formula.apply(inProcessChannel);
            });

    RemoteRetrier retrier = mock(RemoteRetrier.class);
    when(retrier.execute(any()))
        .thenAnswer(
            invocation -> {
              Retrier.RetryableCallable<?, ?> callable = invocation.getArgument(0);
              return callable.call();
            });

    BazelOutputService service =
        new BazelOutputService(
            outputBase,
            () -> execRoot,
            () -> outputPath,
            digestUtil,
            "cache",
            "instance",
            "/prefix",
            maxOutboundMessageSize,
            /* verboseFailures= */ false,
            retrier,
            channel,
            /* lastBuildId= */ null);
    var unused =
        service.startBuild(
            UUID.randomUUID(), "workspace", mock(EventHandler.class), /* finalizeActions= */ true);
    return service;
  }

  @Test
  public void stageArtifacts_smallMessage_singleBatch() throws Exception {
    BazelOutputService service = createService(10000); // large limit
    List<FileMetadata> files = new ArrayList<>();
    for (int i = 0; i < 5; i++) {
      FileMetadata file = mock(FileMetadata.class);
      when(file.path()).thenReturn(outputPath.getRelative("file" + i));
      when(file.digest()).thenReturn(digestUtil.compute(new byte[] {(byte) i}));
      files.add(file);
    }

    service.stageArtifacts(files);

    assertThat(fakeService.getStageRequests()).hasSize(1);
    assertThat(fakeService.getStageRequests().get(0).getArtifactsCount()).isEqualTo(5);
  }

  @Test
  public void stageArtifacts_largeMessage_multipleBatches() throws Exception {
    // Set a small limit to force splitting.
    // We estimate size as: base_size + (entry_size + path_length) * N
    // Entry size without path is estimated with some overhead, let's say it is ~100 bytes.
    // If we set limit to 300, and have paths of length ~10, we should fit about 2-3 artifacts per
    // batch.
    BazelOutputService service = createService(300);
    List<FileMetadata> files = new ArrayList<>();
    for (int i = 0; i < 10; i++) {
      FileMetadata file = mock(FileMetadata.class);
      // Use a relatively long path to increase size
      when(file.path()).thenReturn(outputPath.getRelative("very/long/path/to/file/number/" + i));
      when(file.digest()).thenReturn(digestUtil.compute(new byte[] {(byte) i}));
      files.add(file);
    }

    service.stageArtifacts(files);

    // Verify that we split into multiple requests
    assertThat(fakeService.getStageRequests().size()).isGreaterThan(1);
    int totalStaged = 0;
    for (StageArtifactsRequest request : fakeService.getStageRequests()) {
      totalStaged += request.getArtifactsCount();
      // Verify each request is within limit (approximately, using our estimation)
      // Protobuf serialized size should be within limit.
      assertThat(request.getSerializedSize()).isAtMost(300);
    }
    assertThat(totalStaged).isEqualTo(10);
  }

  @Test
  public void finalizeArtifacts_smallMessage_singleBatch() throws Exception {
    BazelOutputService service = createService(10000); // large limit
    Action action = mock(Action.class);
    List<Artifact> outputs = new ArrayList<>();
    OutputMetadataStore outputMetadataStore = mock(OutputMetadataStore.class);

    for (int i = 0; i < 5; i++) {
      Artifact artifact = ActionsTestUtil.createArtifact(artifactRoot, "out" + i);
      outputs.add(artifact);

      FileArtifactValue metadata = mock(FileArtifactValue.class);
      when(metadata.getType()).thenReturn(FileStateType.REGULAR_FILE);
      when(metadata.getDigest()).thenReturn(new byte[] {(byte) i});
      when(metadata.getSize()).thenReturn(100L);
      when(outputMetadataStore.getOutputMetadata(artifact)).thenReturn(metadata);
    }
    when(action.getOutputs()).thenReturn(outputs);

    service.finalizeAction(action, outputMetadataStore);

    assertThat(fakeService.getFinalizeRequests()).hasSize(1);
    assertThat(fakeService.getFinalizeRequests().get(0).getArtifactsCount()).isEqualTo(5);
  }

  @Test
  public void finalizeArtifacts_largeMessage_multipleBatches() throws Exception {
    BazelOutputService service = createService(300); // small limit
    Action action = mock(Action.class);
    List<Artifact> outputs = new ArrayList<>();
    OutputMetadataStore outputMetadataStore = mock(OutputMetadataStore.class);

    for (int i = 0; i < 10; i++) {
      Artifact artifact =
          ActionsTestUtil.createArtifact(artifactRoot, "very/long/path/to/output/file/number/" + i);
      outputs.add(artifact);

      FileArtifactValue metadata = mock(FileArtifactValue.class);
      when(metadata.getType()).thenReturn(FileStateType.REGULAR_FILE);
      when(metadata.getDigest()).thenReturn(new byte[] {(byte) i});
      when(metadata.getSize()).thenReturn(100L);
      when(outputMetadataStore.getOutputMetadata(artifact)).thenReturn(metadata);
    }
    when(action.getOutputs()).thenReturn(outputs);

    service.finalizeAction(action, outputMetadataStore);

    assertThat(fakeService.getFinalizeRequests().size()).isGreaterThan(1);
    int totalFinalized = 0;
    for (FinalizeArtifactsRequest request : fakeService.getFinalizeRequests()) {
      totalFinalized += request.getArtifactsCount();
      assertThat(request.getSerializedSize()).isAtMost(300);
    }
    assertThat(totalFinalized).isEqualTo(10);
  }

  @Test
  public void finalizeArtifacts_giantArtifact_noEmptyRequest() throws Exception {
    // Limit is small (150), but the base request size (with buildId) plus one giant artifact
    // will definitely exceed it.
    // We want to verify that we don't send an empty request first.
    BazelOutputService service = createService(150);
    Action action = mock(Action.class);
    List<Artifact> outputs = new ArrayList<>();
    OutputMetadataStore outputMetadataStore = mock(OutputMetadataStore.class);

    // Add one giant artifact
    Artifact artifact =
        ActionsTestUtil.createArtifact(
            artifactRoot, "a/very/" + "long/".repeat(20) + "path/to/giant/artifact");
    outputs.add(artifact);

    FileArtifactValue metadata = mock(FileArtifactValue.class);
    when(metadata.getType()).thenReturn(FileStateType.REGULAR_FILE);
    when(metadata.getDigest()).thenReturn(new byte[] {1, 2, 3});
    when(metadata.getSize()).thenReturn(100L);
    when(outputMetadataStore.getOutputMetadata(artifact)).thenReturn(metadata);

    when(action.getOutputs()).thenReturn(outputs);

    service.finalizeAction(action, outputMetadataStore);

    // We expect exactly 1 request (the one containing the giant artifact).
    assertThat(fakeService.getFinalizeRequests()).hasSize(1);
    assertThat(fakeService.getFinalizeRequests().get(0).getArtifactsCount()).isEqualTo(1);
  }
}

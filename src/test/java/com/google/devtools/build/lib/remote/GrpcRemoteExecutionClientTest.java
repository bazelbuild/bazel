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
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.base.Preconditions;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.exec.SpawnResult;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionPolicy;
import com.google.devtools.build.lib.remote.RemoteProtocol.ActionResult;
import com.google.devtools.build.lib.remote.RemoteProtocol.BlobChunk;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadBlobRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadTreeMetadataReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadTreeMetadataRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasLookupReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasLookupRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasStatus;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadBlobReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadBlobRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadTreeMetadataReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadTreeMetadataRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ContentDigest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecuteReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecuteRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheStatus;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionStatus;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.protobuf.ByteString;
import io.grpc.stub.StreamObserver;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for {@link RemoteSpawnRunner} in combination with {@link GrpcRemoteExecutor}. */
@RunWith(JUnit4.class)
public class GrpcRemoteExecutionClientTest {
  private static class MockOwner implements ActionExecutionMetadata {
    private final String mnemonic;
    private final String progressMessage;

    MockOwner(String mnemonic, String progressMessage) {
      this.mnemonic = mnemonic;
      this.progressMessage = progressMessage;
    }

    @Override
    public ActionOwner getOwner() {
      return mock(ActionOwner.class);
    }

    @Override
    public String getMnemonic() {
      return mnemonic;
    }

    @Override
    public String getProgressMessage() {
      return progressMessage;
    }

    @Override
    public boolean inputsDiscovered() {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean discoversInputs() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Iterable<Artifact> getTools() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Iterable<Artifact> getInputs() {
      throw new UnsupportedOperationException();
    }

    @Override
    public RunfilesSupplier getRunfilesSupplier() {
      throw new UnsupportedOperationException();
    }

    @Override
    public ImmutableSet<Artifact> getOutputs() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Iterable<String> getClientEnvironmentVariables() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Artifact getPrimaryInput() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Artifact getPrimaryOutput() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Iterable<Artifact> getMandatoryInputs() {
      throw new UnsupportedOperationException();
    }

    @Override
    public String getKey() {
      throw new UnsupportedOperationException();
    }

    @Override
    public String describeKey() {
      throw new UnsupportedOperationException();
    }

    @Override
    public String prettyPrint() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Iterable<Artifact> getInputFilesForExtraAction(
        ActionExecutionContext actionExecutionContext) {
      return ImmutableList.<Artifact>of();
    }

    @Override
    public ImmutableSet<Artifact> getMandatoryOutputs() {
      throw new UnsupportedOperationException();
    }

    @Override
    public MiddlemanType getActionType() {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean shouldReportPathPrefixConflict(ActionAnalysisMetadata action) {
      throw new UnsupportedOperationException();
    }
  }

  private static final class FakeCas implements GrpcCasInterface {
    private final Map<ByteString, ByteString> content = new HashMap<>();

    public ContentDigest put(byte[] data) {
      ContentDigest digest = ContentDigests.computeDigest(data);
      ByteString key = digest.getDigest();
      ByteString value = ByteString.copyFrom(data);
      content.put(key, value);
      return digest;
    }

    @Override
    public CasLookupReply lookup(CasLookupRequest request) {
      CasStatus.Builder result = CasStatus.newBuilder();
      for (ContentDigest digest : request.getDigestList()) {
        ByteString key = digest.getDigest();
        if (!content.containsKey(key)) {
          result.addMissingDigest(digest);
        }
      }
      if (result.getMissingDigestCount() != 0) {
        result.setError(CasStatus.ErrorCode.MISSING_DIGEST);
      } else {
        result.setSucceeded(true);
      }
      return CasLookupReply.newBuilder().setStatus(result).build();
    }

    @Override
    public CasUploadTreeMetadataReply uploadTreeMetadata(CasUploadTreeMetadataRequest request) {
      return CasUploadTreeMetadataReply.newBuilder()
          .setStatus(CasStatus.newBuilder().setSucceeded(true))
          .build();
    }

    @Override
    public CasDownloadTreeMetadataReply downloadTreeMetadata(
        CasDownloadTreeMetadataRequest request) {
      throw new UnsupportedOperationException();
    }

    @Override
    public Iterator<CasDownloadReply> downloadBlob(CasDownloadBlobRequest request) {
      List<CasDownloadReply> result = new ArrayList<>();
      for (ContentDigest digest : request.getDigestList()) {
        CasDownloadReply.Builder builder = CasDownloadReply.newBuilder();
        ByteString item = content.get(digest.getDigest());
        if (item != null) {
          builder.setStatus(CasStatus.newBuilder().setSucceeded(true));
          builder.setData(BlobChunk.newBuilder().setData(item).setDigest(digest));
        } else {
          throw new IllegalStateException();
        }
        result.add(builder.build());
      }
      return result.iterator();
    }

    @Override
    public StreamObserver<CasUploadBlobRequest> uploadBlobAsync(
        final StreamObserver<CasUploadBlobReply> responseObserver) {
      return new StreamObserver<CasUploadBlobRequest>() {
        private ContentDigest digest;
        private ByteArrayOutputStream current;

        @Override
        public void onNext(CasUploadBlobRequest value) {
          BlobChunk chunk = value.getData();
          if (chunk.hasDigest()) {
            Preconditions.checkState(digest == null);
            digest = chunk.getDigest();
            current = new ByteArrayOutputStream();
          }
          try {
            current.write(chunk.getData().toByteArray());
          } catch (IOException e) {
            throw new RuntimeException(e);
          }
          responseObserver.onNext(
              CasUploadBlobReply.newBuilder()
                  .setStatus(CasStatus.newBuilder().setSucceeded(true))
                  .build());
        }

        @Override
        public void onError(Throwable t) {
          throw new RuntimeException(t);
        }

        @Override
        public void onCompleted() {
          ContentDigest check = ContentDigests.computeDigest(current.toByteArray());
          Preconditions.checkState(check.equals(digest), "%s != %s", digest, check);
          ByteString key = digest.getDigest();
          ByteString value = ByteString.copyFrom(current.toByteArray());
          digest = null;
          current = null;
          content.put(key, value);
          responseObserver.onCompleted();
        }
      };
    }
  }

  private static final class FakeActionInputFileCache implements ActionInputFileCache {
    private final Path execRoot;
    private final BiMap<ActionInput, ByteString> cas = HashBiMap.create();

    FakeActionInputFileCache(Path execRoot) {
      this.execRoot = execRoot;
    }

    void setDigest(ActionInput input, ByteString digest) {
      cas.put(input, digest);
    }

    @Override
    @Nullable
    public byte[] getDigest(ActionInput input) throws IOException {
      return Preconditions.checkNotNull(cas.get(input), input).toByteArray();
    }

    @Override
    public boolean isFile(Artifact input) {
      return execRoot.getRelative(input.getExecPath()).isFile();
    }

    @Override
    public long getSizeInBytes(ActionInput input) throws IOException {
      return execRoot.getRelative(input.getExecPath()).getFileSize();
    }

    @Override
    public boolean contentsAvailableLocally(ByteString digest) {
      throw new UnsupportedOperationException();
    }

    @Override
    @Nullable
    public ActionInput getInputFromDigest(ByteString hexDigest) {
      HashCode code =
          HashCode.fromString(new String(hexDigest.toByteArray(), StandardCharsets.UTF_8));
      ByteString digest = ByteString.copyFrom(code.asBytes());
      return Preconditions.checkNotNull(cas.inverse().get(digest));
    }

    @Override
    public Path getInputPath(ActionInput input) {
      throw new UnsupportedOperationException();
    }
  }

  private static final ArtifactExpander SIMPLE_ARTIFACT_EXPANDER = new ArtifactExpander() {
    @Override
    public void expand(Artifact artifact, Collection<? super Artifact> output) {
      output.add(artifact);
    }
  };

  private FileSystem fs;
  private Path execRoot;
  private EventBus eventBus;
  private SimpleSpawn simpleSpawn;
  private FakeActionInputFileCache fakeFileCache;

  private FileOutErr outErr;
  private long timeoutMillis = 0;

  private final SpawnExecutionPolicy simplePolicy = new SpawnExecutionPolicy() {
    @Override
    public boolean shouldPrefetchInputsForLocalExecution(Spawn spawn) {
      throw new UnsupportedOperationException();
    }

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
      return timeoutMillis;
    }

    @Override
    public FileOutErr getFileOutErr() {
      return outErr;
    }

    @Override
    public SortedMap<PathFragment, ActionInput> getInputMapping() throws IOException {
      return new SpawnInputExpander(/*strict*/false)
          .getInputMapping(simpleSpawn, SIMPLE_ARTIFACT_EXPANDER, fakeFileCache, "workspace");
    }
  };

  @Before
  public final void setUp() throws Exception {
    fs = new InMemoryFileSystem();
    execRoot = fs.getPath("/exec/root");
    FileSystemUtils.createDirectoryAndParents(execRoot);
    eventBus = new EventBus();
    fakeFileCache = new FakeActionInputFileCache(execRoot);
    simpleSpawn = new SimpleSpawn(
        new MockOwner("Mnemonic", "Progress Message"),
        ImmutableList.of("/bin/echo", "Hi!"),
        ImmutableMap.of("VARIABLE", "value"),
        /*executionInfo=*/ImmutableMap.<String, String>of(),
        /*inputs=*/ImmutableList.of(ActionInputHelper.fromPath("input")),
        /*outputs=*/ImmutableList.<ActionInput>of(),
        ResourceSet.ZERO
    );

    Path stdout = fs.getPath("/tmp/stdout");
    Path stderr = fs.getPath("/tmp/stderr");
    FileSystemUtils.createDirectoryAndParents(stdout.getParentDirectory());
    FileSystemUtils.createDirectoryAndParents(stderr.getParentDirectory());
    outErr = new FileOutErr(stdout, stderr);
  }

  private void scratch(ActionInput input, String content) throws IOException {
    Path inputFile = execRoot.getRelative(input.getExecPath());
    FileSystemUtils.writeContentAsLatin1(inputFile, content);
    fakeFileCache.setDigest(
        simpleSpawn.getInputFiles().get(0), ByteString.copyFrom(inputFile.getSHA1Digest()));
  }

  @Test
  public void cacheHit() throws Exception {
    GrpcCasInterface casIface = Mockito.mock(GrpcCasInterface.class);
    GrpcExecutionCacheInterface cacheIface = Mockito.mock(GrpcExecutionCacheInterface.class);
    GrpcExecutionInterface executionIface = Mockito.mock(GrpcExecutionInterface.class);
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    GrpcRemoteExecutor executor =
        new GrpcRemoteExecutor(options, casIface, cacheIface, executionIface);
    RemoteSpawnRunner client =
        new RemoteSpawnRunner(execRoot, eventBus, options, executor);

    scratch(simpleSpawn.getInputFiles().get(0), "xyz");

    ExecutionCacheReply reply = ExecutionCacheReply.newBuilder()
        .setStatus(ExecutionCacheStatus.newBuilder().setSucceeded(true))
        .setResult(ActionResult.newBuilder().setReturnCode(0))
        .build();
    when(cacheIface.getCachedResult(any(ExecutionCacheRequest.class))).thenReturn(reply);

    SpawnResult result = client.exec(simpleSpawn, simplePolicy);
    verify(cacheIface).getCachedResult(any(ExecutionCacheRequest.class));
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(outErr.hasRecordedOutput()).isFalse();
    assertThat(outErr.hasRecordedStderr()).isFalse();
  }

  @Test
  public void cacheHitWithOutput() throws Exception {
    FakeCas casIface = new FakeCas();
    GrpcExecutionCacheInterface cacheIface = Mockito.mock(GrpcExecutionCacheInterface.class);
    GrpcExecutionInterface executionIface = Mockito.mock(GrpcExecutionInterface.class);
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    GrpcRemoteExecutor executor =
        new GrpcRemoteExecutor(options, casIface, cacheIface, executionIface);
    RemoteSpawnRunner client =
        new RemoteSpawnRunner(execRoot, eventBus, options, executor);

    scratch(simpleSpawn.getInputFiles().get(0), "xyz");
    byte[] cacheStdOut = "stdout".getBytes(StandardCharsets.UTF_8);
    byte[] cacheStdErr = "stderr".getBytes(StandardCharsets.UTF_8);
    ContentDigest stdOutDigest = casIface.put(cacheStdOut);
    ContentDigest stdErrDigest = casIface.put(cacheStdErr);

    ExecutionCacheReply reply = ExecutionCacheReply.newBuilder()
        .setStatus(ExecutionCacheStatus.newBuilder().setSucceeded(true))
        .setResult(ActionResult.newBuilder()
            .setReturnCode(0)
            .setStdoutDigest(stdOutDigest)
            .setStderrDigest(stdErrDigest))
        .build();
    when(cacheIface.getCachedResult(any(ExecutionCacheRequest.class))).thenReturn(reply);

    SpawnResult result = client.exec(simpleSpawn, simplePolicy);
    verify(cacheIface).getCachedResult(any(ExecutionCacheRequest.class));
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(outErr.outAsLatin1()).isEqualTo("stdout");
    assertThat(outErr.errAsLatin1()).isEqualTo("stderr");
  }

  @Test
  public void remotelyExecute() throws Exception {
    FakeCas casIface = new FakeCas();
    GrpcExecutionCacheInterface cacheIface = Mockito.mock(GrpcExecutionCacheInterface.class);
    GrpcExecutionInterface executionIface = Mockito.mock(GrpcExecutionInterface.class);
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    GrpcRemoteExecutor executor =
        new GrpcRemoteExecutor(options, casIface, cacheIface, executionIface);
    RemoteSpawnRunner client =
        new RemoteSpawnRunner(execRoot, eventBus, options, executor);

    scratch(simpleSpawn.getInputFiles().get(0), "xyz");
    byte[] cacheStdOut = "stdout".getBytes(StandardCharsets.UTF_8);
    byte[] cacheStdErr = "stderr".getBytes(StandardCharsets.UTF_8);
    ContentDigest stdOutDigest = casIface.put(cacheStdOut);
    ContentDigest stdErrDigest = casIface.put(cacheStdErr);

    ExecutionCacheReply reply = ExecutionCacheReply.newBuilder()
        .setStatus(ExecutionCacheStatus.newBuilder().setSucceeded(true))
        .build();
    when(cacheIface.getCachedResult(any(ExecutionCacheRequest.class))).thenReturn(reply);

    when(executionIface.execute(any(ExecuteRequest.class))).thenReturn(ImmutableList.of(
        ExecuteReply.newBuilder()
            .setStatus(ExecutionStatus.newBuilder().setSucceeded(true))
            .setResult(ActionResult.newBuilder()
                .setReturnCode(0)
                .setStdoutDigest(stdOutDigest)
                .setStderrDigest(stdErrDigest))
            .build()).iterator());

    SpawnResult result = client.exec(simpleSpawn, simplePolicy);
    verify(cacheIface).getCachedResult(any(ExecutionCacheRequest.class));
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(outErr.outAsLatin1()).isEqualTo("stdout");
    assertThat(outErr.errAsLatin1()).isEqualTo("stderr");
  }
}

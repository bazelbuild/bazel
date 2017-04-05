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
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.exec.SpawnResult;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionPolicy;
import com.google.devtools.build.lib.remote.RemoteProtocol.ActionResult;
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
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.SortedMap;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for {@link RemoteSpawnRunner} in combination with {@link GrpcRemoteExecutor}. */
@RunWith(JUnit4.class)
public class GrpcRemoteExecutionClientTest {
  private static final ArtifactExpander SIMPLE_ARTIFACT_EXPANDER = new ArtifactExpander() {
    @Override
    public void expand(Artifact artifact, Collection<? super Artifact> output) {
      output.add(artifact);
    }
  };

  private FileSystem fs;
  private Path execRoot;
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

    @Override
    public void report(ProgressStatus state) {
      // TODO(ulfjack): Test that the right calls are made.
    }
  };

  @Before
  public final void setUp() throws Exception {
    fs = new InMemoryFileSystem();
    execRoot = fs.getPath("/exec/root");
    FileSystemUtils.createDirectoryAndParents(execRoot);
    fakeFileCache = new FakeActionInputFileCache(execRoot);
    simpleSpawn = new SimpleSpawn(
        new FakeOwner("Mnemonic", "Progress Message"),
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
    RemoteSpawnRunner client = new RemoteSpawnRunner(execRoot, options, executor);

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
    InMemoryCas casIface = new InMemoryCas();
    GrpcExecutionCacheInterface cacheIface = Mockito.mock(GrpcExecutionCacheInterface.class);
    GrpcExecutionInterface executionIface = Mockito.mock(GrpcExecutionInterface.class);
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    GrpcRemoteExecutor executor =
        new GrpcRemoteExecutor(options, casIface, cacheIface, executionIface);
    RemoteSpawnRunner client = new RemoteSpawnRunner(execRoot, options, executor);

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
    InMemoryCas casIface = new InMemoryCas();
    GrpcExecutionCacheInterface cacheIface = Mockito.mock(GrpcExecutionCacheInterface.class);
    GrpcExecutionInterface executionIface = Mockito.mock(GrpcExecutionInterface.class);
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    GrpcRemoteExecutor executor =
        new GrpcRemoteExecutor(options, casIface, cacheIface, executionIface);
    RemoteSpawnRunner client = new RemoteSpawnRunner(execRoot, options, executor);

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

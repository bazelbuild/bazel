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
import static org.mockito.Matchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyZeroInteractions;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.exec.SpawnResult;
import com.google.devtools.build.lib.exec.SpawnResult.Status;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionPolicy;
import com.google.devtools.build.lib.exec.util.FakeOwner;
import com.google.devtools.build.lib.remote.Digests.ActionKey;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.devtools.remoteexecution.v1test.ActionResult;
import com.google.devtools.remoteexecution.v1test.ExecuteRequest;
import com.google.devtools.remoteexecution.v1test.ExecuteResponse;
import java.io.IOException;
import java.time.Duration;
import java.util.Collection;
import java.util.SortedMap;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/** Tests for {@link com.google.devtools.build.lib.remote.RemoteSpawnRunner} */
@RunWith(JUnit4.class)
public class RemoteSpawnRunnerTest {

  private static final ImmutableMap<String, String> NO_CACHE =
      ImmutableMap.of(ExecutionRequirements.NO_CACHE, "");

  private Path execRoot;
  private FakeActionInputFileCache fakeFileCache;
  private FileOutErr outErr;

  @Mock
  private RemoteActionCache cache;

  @Mock
  private GrpcRemoteExecutor executor;

  @Mock
  private SpawnRunner localRunner;

  @Before
  public final void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);

    FileSystem fs = new InMemoryFileSystem();
    execRoot = fs.getPath("/exec/root");
    FileSystemUtils.createDirectoryAndParents(execRoot);
    fakeFileCache = new FakeActionInputFileCache(execRoot);

    Path stdout = fs.getPath("/tmp/stdout");
    Path stderr = fs.getPath("/tmp/stderr");
    FileSystemUtils.createDirectoryAndParents(stdout.getParentDirectory());
    FileSystemUtils.createDirectoryAndParents(stderr.getParentDirectory());
    outErr = new FileOutErr(stdout, stderr);
  }

  @Test
  @SuppressWarnings("unchecked")
  public void nonCachableSpawnsShouldNotBeCached_remote() throws Exception {
    // Test that if a spawn is marked "NO_CACHE" that it's neither fetched from a remote cache
    // nor uploaded to a remote cache. It should be executed remotely, however.

    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteAcceptCached = true;
    options.remoteLocalFallback = false;
    options.remoteUploadLocalResults = true;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(execRoot, options, localRunner, true, cache, executor);

    ExecuteResponse succeeded = ExecuteResponse.newBuilder().setResult(
        ActionResult.newBuilder().setExitCode(0).build()).build();
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenReturn(succeeded);

    Spawn spawn = new SimpleSpawn(
        new FakeOwner("foo", "bar"),
        /*arguments=*/ ImmutableList.of(),
        /*environment=*/ ImmutableMap.of(),
        NO_CACHE,
        /*inputs=*/ ImmutableList.of(),
        /*outputs=*/ ImmutableList.<ActionInput>of(),
        ResourceSet.ZERO);

    SpawnExecutionPolicy policy = new FakeSpawnExecutionPolicy(spawn);

    runner.exec(spawn, policy);

    ArgumentCaptor<ExecuteRequest> requestCaptor = ArgumentCaptor.forClass(ExecuteRequest.class);
    verify(executor).executeRemotely(requestCaptor.capture());
    assertThat(requestCaptor.getValue().getSkipCacheLookup()).isTrue();

    verify(cache, never())
        .getCachedActionResult(any(ActionKey.class));
    verify(cache, never()).upload(any(ActionKey.class), any(Path.class), any(Collection.class),
          any(FileOutErr.class));
    verifyZeroInteractions(localRunner);
  }

  @Test
  @SuppressWarnings("unchecked")
  public void nonCachableSpawnsShouldNotBeCached_local() throws Exception {
    // Test that if a spawn is executed locally, due to the local fallback, that its result is not
    // uploaded to the remote cache.

    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteAcceptCached = true;
    options.remoteLocalFallback = true;
    options.remoteUploadLocalResults = true;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(execRoot, options, localRunner, true, cache, null);

    // Throw an IOException to trigger the local fallback.
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenThrow(IOException.class);

    Spawn spawn = new SimpleSpawn(
        new FakeOwner("foo", "bar"),
        /*arguments=*/ ImmutableList.of(),
        /*environment=*/ ImmutableMap.of(),
        NO_CACHE,
        /*inputs=*/ ImmutableList.of(),
        /*outputs=*/ ImmutableList.<ActionInput>of(),
        ResourceSet.ZERO);

    SpawnExecutionPolicy policy = new FakeSpawnExecutionPolicy(spawn);

    runner.exec(spawn, policy);

    verify(localRunner).exec(spawn, policy);

    verify(cache, never())
        .getCachedActionResult(any(ActionKey.class));
    verify(cache, never()).upload(any(ActionKey.class), any(Path.class), any(Collection.class),
        any(FileOutErr.class));
  }

  @Test
  @SuppressWarnings("unchecked")
  public void failedActionShouldNotBeUploaded() throws Exception {
    // Test that the outputs of a failed locally executed action are not uploaded to a remote
    // cache.

    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteUploadLocalResults = true;

    RemoteSpawnRunner runner =
        spy(new RemoteSpawnRunner(execRoot, options, localRunner, true, cache, null));

    Spawn spawn = new SimpleSpawn(
        new FakeOwner("foo", "bar"),
        /*arguments=*/ ImmutableList.of(),
        /*environment=*/ ImmutableMap.of(),
        /*executionInfo=*/ ImmutableMap.of(),
        /*inputs=*/ ImmutableList.of(),
        /*outputs=*/ ImmutableList.<ActionInput>of(),
        ResourceSet.ZERO);
    SpawnExecutionPolicy policy = new FakeSpawnExecutionPolicy(spawn);

    SpawnResult res = Mockito.mock(SpawnResult.class);
    when(res.exitCode()).thenReturn(1);
    when(res.status()).thenReturn(Status.EXECUTION_FAILED);
    when(localRunner.exec(eq(spawn), eq(policy))).thenReturn(res);

    assertThat(runner.exec(spawn, policy)).isSameAs(res);

    verify(localRunner).exec(eq(spawn), eq(policy));
    verify(runner).execLocallyAndUpload(eq(spawn), eq(policy), any(SortedMap.class), eq(cache),
        any(ActionKey.class));
    verify(cache, never()).upload(any(ActionKey.class), any(Path.class), any(Collection.class),
        any(FileOutErr.class));
  }

  // TODO(buchgr): Extract a common class to be used for testing.
  class FakeSpawnExecutionPolicy implements SpawnExecutionPolicy {

    private final ArtifactExpander artifactExpander =
        (artifact, output) -> output.add(artifact);

    private final Spawn spawn;

    FakeSpawnExecutionPolicy(Spawn spawn) {
      this.spawn = spawn;
    }

    @Override
    public int getId() {
      return 0;
    }

    @Override
    public void prefetchInputs() throws IOException {
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
    public SortedMap<PathFragment, ActionInput> getInputMapping() throws IOException {
      return new SpawnInputExpander(/*strict*/ false)
          .getInputMapping(spawn, artifactExpander, fakeFileCache, "workspace");
    }

    @Override
    public void report(ProgressStatus state, String name) {
      assertThat(state).isEqualTo(ProgressStatus.EXECUTING);
    }
  }
}

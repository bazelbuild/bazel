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
import static org.mockito.Matchers.any;
import static org.mockito.Matchers.eq;
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
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.util.Collection;
import java.util.SortedMap;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;
import org.mockito.MockitoAnnotations.Mock;

/** Tests for {@link CachedLocalSpawnRunner}. */
@RunWith(JUnit4.class)
public class CachedLocalSpawnRunnerTest {
  private static final ArtifactExpander SIMPLE_ARTIFACT_EXPANDER =
      new ArtifactExpander() {
        @Override
        public void expand(Artifact artifact, Collection<? super Artifact> output) {
          output.add(artifact);
        }
      };

  private FileSystem fs;
  private Path execRoot;
  private SimpleSpawn simpleSpawn;
  private FakeActionInputFileCache fakeFileCache;
  @Mock private RemoteActionCache cache;
  @Mock private SpawnRunner delegate;
  CachedLocalSpawnRunner runner;
  private FileOutErr outErr;

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
    MockitoAnnotations.initMocks(this);
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
    runner = new CachedLocalSpawnRunner(execRoot, options, cache, delegate);
    fakeFileCache.createScratchInput(simpleSpawn.getInputFiles().get(0), "xyz");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void cacheHit() throws Exception {
    ActionResult actionResult = ActionResult.getDefaultInstance();
    when(cache.getCachedActionResult(any(ActionKey.class))).thenReturn(actionResult);

    SpawnResult result = runner.exec(simpleSpawn, simplePolicy);
    // In line with state-based testing, we only verify calls that produce side effects.
    verify(cache).downloadAllResults(actionResult, execRoot);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(outErr.hasRecordedOutput()).isFalse();
    assertThat(outErr.hasRecordedStderr()).isFalse();
  }

  @SuppressWarnings("unchecked")
  @Test
  public void cacheHitWithOutput() throws Exception {
    byte[] cacheStdOut = "stdout".getBytes(UTF_8);
    byte[] cacheStdErr = "stderr".getBytes(UTF_8);
    Digest stdOutDigest = Digests.computeDigest(cacheStdOut);
    Digest stdErrDigest = Digests.computeDigest(cacheStdErr);

    ActionResult actionResult =
        ActionResult.newBuilder()
            .setStdoutDigest(stdOutDigest)
            .setStderrDigest(stdErrDigest)
            .build();
    when(cache.getCachedActionResult(any(ActionKey.class))).thenReturn(actionResult);
    when(cache.downloadBlob(stdOutDigest)).thenReturn(cacheStdOut);
    when(cache.downloadBlob(stdErrDigest)).thenReturn(cacheStdErr);

    SpawnResult result = runner.exec(simpleSpawn, simplePolicy);
    // In line with state-based testing, we only verify calls that produce side effects.
    verify(cache).downloadAllResults(actionResult, execRoot);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(outErr.outAsLatin1()).isEqualTo("stdout");
    assertThat(outErr.errAsLatin1()).isEqualTo("stderr");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void cacheHitWithOutputsInlined() throws Exception {
    ActionResult actionResult =
        ActionResult.newBuilder()
            .setStdoutRaw(ByteString.copyFromUtf8("stdout"))
            .setStderrRaw(ByteString.copyFromUtf8("stderr"))
            .build();
    when(cache.getCachedActionResult(any(ActionKey.class))).thenReturn(actionResult);

    SpawnResult result = runner.exec(simpleSpawn, simplePolicy);
    // In line with state-based testing, we only verify calls that produce side effects.
    verify(cache).downloadAllResults(actionResult, execRoot);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(outErr.outAsLatin1()).isEqualTo("stdout");
    assertThat(outErr.errAsLatin1()).isEqualTo("stderr");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void cacheMiss() throws Exception {
    byte[] cacheStdOut = "stdout".getBytes(UTF_8);
    byte[] cacheStdErr = "stderr".getBytes(UTF_8);
    Digest stdOutDigest = Digests.computeDigest(cacheStdOut);
    Digest stdErrDigest = Digests.computeDigest(cacheStdErr);
    when(cache.uploadFileContents(any(Path.class))).thenReturn(stdErrDigest, stdOutDigest);
    SpawnResult delegateResult =
        new SpawnResult.Builder().setExitCode(0).setStatus(Status.SUCCESS).build();
    when(delegate.exec(any(Spawn.class), any(SpawnExecutionPolicy.class)))
        .thenReturn(delegateResult);

    SpawnResult result = runner.exec(simpleSpawn, simplePolicy);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.exitCode()).isEqualTo(0);
    // We use verify to check that each method is called the correct number of times.
    verify(cache)
        .uploadAllResults(any(Path.class), any(Collection.class), any(ActionResult.Builder.class));
    // Two additional uploads for stdout and stderr.
    verify(cache, Mockito.times(2)).uploadFileContents(any(Path.class));
    ActionResult actionResult =
        ActionResult.newBuilder()
            .setStdoutDigest(stdOutDigest)
            .setStderrDigest(stdErrDigest)
            .build();
    verify(cache).setCachedActionResult(any(ActionKey.class), eq(actionResult));
  }
}

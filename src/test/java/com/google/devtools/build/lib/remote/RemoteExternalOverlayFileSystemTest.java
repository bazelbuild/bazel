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
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.Tree;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import java.io.IOException;
import java.time.Duration;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RemoteExternalOverlayFileSystem}. */
@RunWith(JUnit4.class)
public final class RemoteExternalOverlayFileSystemTest {

  private static final DigestHashFunction HASH_FUNCTION = DigestHashFunction.SHA256;
  private static final Duration REMOTE_CACHE_TTL = Duration.ofDays(1);

  private final DigestUtil digestUtil = new DigestUtil(SyscallCache.NO_CACHE, HASH_FUNCTION);
  private final FileSystem nativeFs = new InMemoryFileSystem(HASH_FUNCTION);
  private final PathFragment externalDir = PathFragment.create("/output_base/external");
  private final RemoteExternalOverlayFileSystem overlayFs =
      new RemoteExternalOverlayFileSystem(externalDir, nativeFs);
  private final AbstractActionInputPrefetcher prefetcher =
      mock(AbstractActionInputPrefetcher.class);
  private final MemoizingEvaluator evaluator = mock(MemoizingEvaluator.class);
  private InMemoryCombinedCache cache;

  @Before
  public void setUp() throws IOException {
    cache = new InMemoryCombinedCache(digestUtil);
    nativeFs.getPath(externalDir).createDirectoryAndParents();
    when(prefetcher.prefetchFilesInterruptibly(any(), any(), any(), any(), any()))
        .thenReturn(immediateVoidFuture());
  }

  /**
   * Regression test for <a href="https://github.com/bazelbuild/bazel/issues/30110">#30110</a>.
   *
   * <p>External repo contents injected by a previous command persist in {@code externalFs}
   * across commands, but the overlay's per-build fields ({@code reporter}, {@code cache}, ...)
   * are cleared in {@link RemoteExternalOverlayFileSystem#afterCommand}. If Skyframe reaches
   * into a still-injected repo before the next {@code beforeCommand} rewires those fields
   * (e.g. from {@code IgnoredSubdirectoriesFunction} loading {@code .bazelignore}), the inner
   * {@code RemoteExternalFileSystem.getInputStream} used to crash Bazel with an NPE because it
   * unconditionally dereferenced the null {@code reporter} and {@code cache}. It should instead
   * surface a clean {@link IOException} so the SkyFunction can fail gracefully.
   */
  @Test
  public void getInputStream_perBuildStateCleared_doesNotCrashWithNpe() throws Exception {
    RepositoryName repo = RepositoryName.createUnvalidated("some_repo+");
    Digest digest = uploadToCache("ignored\n".getBytes(UTF_8));
    // .bazelignore isn't prefetched (only .bzl and REPO.bazel are), so its contents will be
    // fetched on demand from the CombinedCache by RemoteExternalFileSystem.getInputStream.
    Tree tree =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addFiles(
                        FileNode.newBuilder().setName(".bazelignore").setDigest(digest).build())
                    .build())
            .build();

    overlayFs.beforeCommand(
        cache, prefetcher, new Reporter(), "build-req", "cmd", evaluator, REMOTE_CACHE_TTL);
    assertThat(overlayFs.injectRemoteRepo(repo, tree, "marker-contents")).isTrue();
    overlayFs.afterCommand();

    PathFragment injectedFile = externalDir.getChild(repo.getName()).getChild(".bazelignore");
    IOException e =
        assertThrows(IOException.class, () -> overlayFs.getInputStream(injectedFile));
    assertThat(e).isNotInstanceOf(NullPointerException.class);
  }

  private Digest uploadToCache(byte[] contents) throws Exception {
    RemoteActionExecutionContext context =
        RemoteActionExecutionContext.create(
            TracingMetadataUtils.buildMetadata(
                "build-req", "cmd", "action-id", /* actionMetadata= */ null));
    return cache.addContents(context, contents);
  }
}

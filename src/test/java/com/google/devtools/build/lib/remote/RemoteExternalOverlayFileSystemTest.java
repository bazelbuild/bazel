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
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.Tree;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class RemoteExternalOverlayFileSystemTest {
  @Test
  public void ensureMaterialized_previousCallerFinishesBeforeTaskSubmission() throws Exception {
    var digestUtil = new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
    var cache = new InMemoryCombinedCache(digestUtil);
    var nativeFs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    var externalRoot = PathFragment.create("/output/external");
    var overlay = new RemoteExternalOverlayFileSystem(externalRoot, nativeFs);
    var reporter = new Reporter();
    var prefetcher = mock(AbstractActionInputPrefetcher.class);
    when(prefetcher.prefetchFilesInterruptibly(isNull(), any(), any(), any(), any()))
        .thenReturn(immediateVoidFuture());
    overlay.beforeCommand(
        cache,
        prefetcher,
        reporter,
        "build-request",
        "command",
        mock(MemoizingEvaluator.class),
        Duration.ofMinutes(1));
    try {
      var repo = RepositoryName.create("repo");
      assertThat(overlay.injectRemoteRepo(repo, Tree.getDefaultInstance(), "marker")).isTrue();
      var delayedRepo = mock(RepositoryName.class);
      when(delayedRepo.getMarkerFileName()).thenReturn(repo.getMarkerFileName());
      var calls = new AtomicInteger();
      when(delayedRepo.getName())
          .thenAnswer(
              unused -> {
                // Finish another caller after the presence check but before task submission.
                if (calls.incrementAndGet() == 2) {
                  overlay.ensureMaterialized(repo, reporter);
                }
                return repo.getName();
              });

      overlay.ensureMaterialized(delayedRepo, reporter);

      assertThat(
              FileSystemUtils.readContent(
                  nativeFs.getPath(externalRoot.getChild(repo.getMarkerFileName())), UTF_8))
          .isEqualTo("marker");
    } finally {
      overlay.afterCommand();
    }
  }
}

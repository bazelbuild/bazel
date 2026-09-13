// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.includescanning;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.StaticInputMetadataProvider;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RewindableRepoFileSystem;
import com.google.devtools.build.lib.vfs.RewindingSynchronizer;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.concurrent.TimeUnit;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests include discovery while an external source's repository is replaced. */
@RunWith(JUnit4.class)
public final class IncludeParserTest {
  private static final RepositoryName REPO = RepositoryName.createUnvalidated("repo");

  @Test
  public void extractInclusions_waitsForRepoReplacementUsingOriginalFileSystem() throws Exception {
    var fs = new RepoFileSystem();
    var inputFs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    var inputPath = inputFs.getPath("/input.cc");
    Artifact artifact =
        new Artifact.SourceArtifact(
            ArtifactRoot.asExternalSourceRoot(Root.fromPath(fs.getPath("/output/external/repo"))),
            PathFragment.create("external/repo/input.cc"),
            () -> Label.parseCanonicalUnchecked("@@repo//:input.cc"));
    var context = mock(ActionExecutionContext.class);
    when(context.getInputPath(artifact)).thenReturn(inputPath);
    when(context.getInputMetadataProvider())
        .thenReturn(new StaticInputMetadataProvider(ImmutableMap.of()));

    // The execution filesystem's path is absent until the fetch finishes. The lock must be taken
    // through the artifact's original filesystem, even when the read uses another one.
    assertReadWaitsForRepoReplacement(fs, artifact, context, inputPath);
  }

  @Test
  public void extractInclusions_symlinkToRepoFile_waitsForRepoReplacement() throws Exception {
    var fs = new RepoFileSystem();
    Artifact artifact =
        ActionsTestUtil.createArtifact(
            ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out"), "input.cc");
    var inputPath = artifact.getPath();
    var context = mock(ActionExecutionContext.class);
    when(context.getInputPath(artifact)).thenReturn(inputPath);
    // The artifact was materialized as a symlink to a file in the repository, e.g. by a symlink
    // action, so reading it reads that repository.
    when(context.getInputMetadataProvider())
        .thenReturn(
            new StaticInputMetadataProvider(
                ImmutableMap.of(
                    artifact,
                    FileArtifactValue.createFromExistingWithResolvedPath(
                        FileArtifactValue.createForRemoteFile(new byte[] {1}, 1, 1),
                        PathFragment.create("/output/external/repo/input.cc")))));

    assertReadWaitsForRepoReplacement(fs, artifact, context, inputPath);
  }

  private static void assertReadWaitsForRepoReplacement(
      RepoFileSystem fs, Artifact artifact, ActionExecutionContext context, Path inputPath)
      throws Exception {
    fs.synchronizer.markReplacementsPossible();
    var reader =
        new TestThread(
            () -> {
              var inclusions =
                  new IncludeParser(null)
                      .extractInclusions(artifact, null, context, null, null, null, false);
              assertThat(inclusions).hasSize(1);
            });
    try (var unused = fs.synchronizer.acquireWriteLock(REPO)) {
      reader.start();
      long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
      while (reader.getState() != Thread.State.WAITING) {
        assertThat(reader.isAlive()).isTrue();
        assertThat(System.nanoTime()).isLessThan(deadline);
        Thread.sleep(1);
      }
      inputPath.getParentDirectory().createDirectoryAndParents();
      FileSystemUtils.writeContent(inputPath, UTF_8, "#include \"restored.h\"\n");
    }
    reader.joinAndAssertState(10_000);
  }

  private static final class RepoFileSystem extends InMemoryFileSystem
      implements RewindableRepoFileSystem {
    private static final PathFragment EXTERNAL_DIR = PathFragment.create("/output/external");

    private final RewindingSynchronizer synchronizer = new RewindingSynchronizer();

    RepoFileSystem() {
      super(DigestHashFunction.SHA256);
    }

    @Override
    public RewindingSynchronizer getRewindingSynchronizer() {
      return synchronizer;
    }

    @Override
    public boolean isRepoPath(PathFragment path) {
      return path.startsWith(EXTERNAL_DIR) && path.segmentCount() > EXTERNAL_DIR.segmentCount();
    }

    @Override
    public RepositoryName repoContaining(PathFragment path) {
      return RepositoryName.createUnvalidated(path.getSegment(EXTERNAL_DIR.segmentCount()));
    }
  }
}

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
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileContentsProxy;
import com.google.devtools.build.lib.actions.cache.OutputMetadataStore;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.remote.disk.DiskCacheClient;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.io.InputStream;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;

/** Checks that deduplication preserves action metadata without rereading output contents. */
@RunWith(JUnit4.class)
public final class DiskCacheOutputDeduplicatorTest {
  private final class TrackingFileSystem extends InMemoryFileSystem {
    boolean rejectOutputReads;

    TrackingFileSystem() {
      super(DigestHashFunction.SHA256);
    }

    @Override
    public synchronized InputStream getInputStream(PathFragment path) throws IOException {
      if (rejectOutputReads && path.startsWith(PathFragment.create("/exec/outputs"))) {
        throw new AssertionError("Output contents were read again: " + path);
      }
      return super.getInputStream(path);
    }
  }

  private final TrackingFileSystem fs = new TrackingFileSystem();
  private final DigestUtil digests =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
  private final DiskCacheClient cache;
  private final DiskCacheOutputDeduplicator deduplicator;
  private final ArtifactRoot root =
      ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "outputs");

  public DiskCacheOutputDeduplicatorTest() throws IOException {
    cache = new DiskCacheClient(fs.getPath("/cache"), digests, true);
    deduplicator = new DiskCacheOutputDeduplicator(cache, mock(EventHandler.class));
  }

  @After
  public void tearDown() {
    cache.close();
  }

  private FileArtifactValue createOutput(Path output, String content, boolean seedCache)
      throws Exception {
    output.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContent(output, UTF_8, content);
    var digest = digests.compute(content.getBytes(UTF_8));
    if (seedCache) {
      var unused = getFromFuture(cache.uploadFile(digest, output));
    }
    output.chmod(0555);
    return FileArtifactValue.createForNormalFile(
        digests.hash(content.getBytes(UTF_8)),
        FileContentsProxy.create(output.stat()),
        content.getBytes(UTF_8).length);
  }

  @Test
  public void replacesContentsProxyWithoutReadingOutputAgain() throws Exception {
    Path output = fs.getPath("/exec/outputs/file");
    var original = createOutput(output, "contents", true);
    long mtime = output.getLastModifiedTime();
    fs.rejectOutputReads = true;

    var result = deduplicator.deduplicate(output, original);

    assertThat(result).isNotSameInstanceAs(original);
    assertThat(result.getDigest()).isEqualTo(original.getDigest());
    assertThat(result.getSize()).isEqualTo(original.getSize());
    assertThat(result.getContentsProxy()).isNotEqualTo(original.getContentsProxy());
    assertThat(result.wasModifiedSinceDigest(output)).isFalse();
    assertThat(output.getLastModifiedTime()).isEqualTo(mtime);
    assertThat(output.stat().getPermissions()).isEqualTo(0555);
    fs.rejectOutputReads = false;
    assertThat(FileSystemUtils.readContent(output, UTF_8)).isEqualTo("contents");
  }

  @Test
  public void missingCacheDoesNotReplaceOutput() throws Exception {
    Path output = fs.getPath("/exec/outputs/file");
    var original = createOutput(output, "contents", false);
    fs.rejectOutputReads = true;

    assertThat(deduplicator.deduplicate(output, original)).isSameInstanceAs(original);
    assertThat(original.wasModifiedSinceDigest(output)).isFalse();
  }

  @Test
  public void changedOutputIsNotReplaced() throws Exception {
    Path output = fs.getPath("/exec/outputs/file");
    var original = createOutput(output, "contents", true);
    output.chmod(0755);
    FileSystemUtils.writeContent(output, UTF_8, "modified");
    output.setLastModifiedTime(output.getLastModifiedTime() + 1000000000L);
    fs.rejectOutputReads = true;

    assertThat(deduplicator.deduplicate(output, original)).isSameInstanceAs(original);
    fs.rejectOutputReads = false;
    assertThat(FileSystemUtils.readContent(output, UTF_8)).isEqualTo("modified");
  }

  @Test
  public void remoteOutputIsNotMaterialized() throws Exception {
    Path output = fs.getPath("/exec/outputs/absent");
    var digest = digests.compute("contents".getBytes(UTF_8));
    var remote =
        FileArtifactValue.createForRemoteFile(
            digests.hash("contents".getBytes(UTF_8)), digest.getSizeBytes(), 0);

    assertThat(deduplicator.deduplicate(output, remote)).isSameInstanceAs(remote);
    assertThat(output.exists()).isFalse();
  }

  @Test
  public void emptyOutputIsNotReplaced() throws Exception {
    Path output = fs.getPath("/exec/outputs/file");
    var original = createOutput(output, "", true);

    assertThat(deduplicator.deduplicate(output, original)).isSameInstanceAs(original);
  }

  @Test
  public void finalizeActionInjectsNewRegularMetadata() throws Exception {
    Artifact output = ActionsTestUtil.createArtifact(root, "file");
    var original = createOutput(output.getPath(), "contents", true);
    var action = mock(Action.class);
    when(action.getOutputs()).thenReturn(ImmutableList.of(output));
    var store = mock(OutputMetadataStore.class);
    when(store.getOutputMetadata(output)).thenReturn(original);
    fs.rejectOutputReads = true;

    deduplicator.finalizeAction(action, store);

    var metadata = ArgumentCaptor.forClass(FileArtifactValue.class);
    verify(store).injectFile(eq(output), metadata.capture());
    assertThat(metadata.getValue().wasModifiedSinceDigest(output.getPath())).isFalse();
  }

  @Test
  public void treeChildrenKeepDigestAndReadonlyParent() throws Exception {
    var parent = ActionsTestUtil.createTreeArtifactWithGeneratingAction(root, "tree");
    var child = TreeFileArtifact.createTreeOutput(parent, "file");
    var original = createOutput(child.getPath(), "contents", true);
    parent.getPath().chmod(0555);
    var tree = TreeArtifactValue.newBuilder(parent).putChild(child, original).build();
    var action = mock(Action.class);
    when(action.getOutputs()).thenReturn(ImmutableList.of(parent));
    var store = mock(OutputMetadataStore.class);
    when(store.getTreeArtifactValue(parent)).thenReturn(tree);
    fs.rejectOutputReads = true;

    deduplicator.finalizeAction(action, store);

    var metadata = ArgumentCaptor.forClass(TreeArtifactValue.class);
    verify(store).injectTree(eq(parent), metadata.capture());
    assertThat(metadata.getValue().getDigest()).isEqualTo(tree.getDigest());
    assertThat(
            metadata.getValue().getChildValues().get(child).wasModifiedSinceDigest(child.getPath()))
        .isFalse();
    assertThat(parent.getPath().stat().getPermissions()).isEqualTo(0555);
    assertThat(parent.getPath().getDirectoryEntries()).containsExactly(child.getPath());
  }
}

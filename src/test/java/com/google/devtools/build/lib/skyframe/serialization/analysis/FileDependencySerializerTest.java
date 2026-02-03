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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.skyframe.FileKey;
import com.google.devtools.build.lib.skyframe.serialization.KeyValueWriter;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.FileDataInfoOrFuture;
import com.google.devtools.build.lib.skyframe.serialization.analysis.InvalidationDataInfoOrFuture.FutureFileDataInfo;
import com.google.devtools.build.lib.versioning.LongVersionGetter;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.InMemoryNodeEntry;
import java.util.concurrent.ExecutionException;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

@RunWith(JUnit4.class)
public final class FileDependencySerializerTest {

  @Rule public final MockitoRule mocks = MockitoJUnit.rule();
  @Mock private LongVersionGetter versionGetter;
  @Mock private InMemoryGraph graph;
  @Mock private KeyValueWriter writer;
  @Mock private InMemoryNodeEntry nodeEntry;

  private FileDependencySerializer serializer;
  private Root root;

  @Before
  public void setUp() throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    root = Root.fromPath(fs.getPath("/root"));
    root.asPath().createDirectoryAndParents();
    serializer = new FileDependencySerializer(versionGetter, graph, writer);
  }

  @Test
  public void missingNodeEntry_incrementsErrorCounter() {
    FileKey key = FileKey.create(RootedPath.toRootedPath(root, PathFragment.create("missing.txt")));
    when(graph.getIfPresent(key)).thenReturn(null);

    FileDataInfoOrFuture result = serializer.registerDependency(key);

    assertThat(result).isInstanceOf(FutureFileDataInfo.class);
    ExecutionException e =
        assertThrows(ExecutionException.class, () -> ((FutureFileDataInfo) result).get());
    assertThat(e).hasCauseThat().isInstanceOf(MissingSkyframeEntryException.class);
    assertThat(serializer.getCounters().nodesWithProcessingErrors.get()).isEqualTo(1);
    assertThat(serializer.getCounters().nodesWaitingForDeps.get()).isEqualTo(0);
  }

  @Test
  public void rootDirectoryDependency_isConstantAndDecrementsCounter() throws Exception {
    FileKey key = FileKey.create(RootedPath.toRootedPath(root, PathFragment.EMPTY_FRAGMENT));

    FileDataInfoOrFuture result = serializer.registerDependency(key);

    assertThat(result).isEqualTo(InvalidationDataInfoOrFuture.ConstantFileData.CONSTANT_FILE);
    assertThat(serializer.getCounters().nodesWaitingForDeps.get()).isEqualTo(0);
    assertThat(serializer.getCounters().nodesWithProcessingErrors.get()).isEqualTo(0);
  }

  @Test
  public void symlinkResolutionFailure_incrementsErrorCounter() throws Exception {
    PathFragment symlinkPathFragment = PathFragment.create("symlink.txt");
    PathFragment targetPathFragment = PathFragment.create("target.txt");
    RootedPath symlinkRootedPath = RootedPath.toRootedPath(root, symlinkPathFragment);
    FileKey symlinkKey = FileKey.create(symlinkRootedPath);

    FileValue symlinkFsv = mock(FileValue.class);
    when(symlinkFsv.isSymlink()).thenReturn(true);
    when(symlinkFsv.getUnresolvedLinkTarget()).thenReturn(targetPathFragment);
    when(symlinkFsv.realRootedPath(symlinkRootedPath)).thenReturn(symlinkRootedPath);
    when(symlinkFsv.exists()).thenReturn(true);
    when(symlinkFsv.isDirectory()).thenReturn(false);
    when(nodeEntry.getValue()).thenReturn(symlinkFsv);
    when(graph.getIfPresent(symlinkKey)).thenReturn(nodeEntry);

    // Symlink resolution calls getVersion on link path.
    when(versionGetter.getFilePathOrSymlinkVersion(symlinkRootedPath.asPath())).thenReturn(2L);

    // Create the failure mode where the symlink target does not exist in graph.
    RootedPath targetRootedPath = RootedPath.toRootedPath(root, targetPathFragment);
    when(graph.getIfPresent(targetRootedPath)).thenReturn(null);

    FileDataInfoOrFuture result = serializer.registerDependency(symlinkKey);

    assertThat(result).isInstanceOf(FutureFileDataInfo.class);
    ExecutionException e =
        assertThrows(ExecutionException.class, () -> ((FutureFileDataInfo) result).get());
    assertThat(e).hasCauseThat().isInstanceOf(MissingSkyframeEntryException.class);
    assertThat(serializer.getCounters().nodesWithProcessingErrors.get()).isEqualTo(1);
    assertThat(serializer.getCounters().nodesWaitingForDeps.get()).isEqualTo(0);
  }
}

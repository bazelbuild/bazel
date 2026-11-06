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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.DirectoryListingKey;
import com.google.devtools.build.lib.skyframe.DirectoryListingValue;
import com.google.devtools.build.lib.skyframe.FileKey;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNode;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class LeafIdAssignmentTest {

  private final Root root =
      Root.fromPath(new InMemoryFileSystem(DigestHashFunction.SHA256).getPath("/root"));

  private final FileKey file = FileKey.create(rootedPath("a.txt"));
  private final DirectoryListingKey listing = DirectoryListingValue.key(rootedPath("dir"));

  @Test
  public void dense_returnsAssignedIds() {
    LeafIdAssignment ids = dense(/* fileId= */ 0, /* listingId= */ 1, /* sourceId= */ 2);

    assertThat(ids.getAnalysisId(file)).isEqualTo(0);
    assertThat(ids.getAnalysisId(listing)).isEqualTo(1);
    assertThat(ids.getSourceId(file)).isEqualTo(2);
    assertThat(ids.denseIdCount()).isEqualTo(3);
    assertThat(ids.fallbackIdCount()).isEqualTo(0);
  }

  @Test
  public void dense_missingLeaf_failsFast() {
    var ids = LeafIdAssignment.dense(new Object2IntOpenHashMap<>(), new Object2IntOpenHashMap<>());

    assertThat(assertThrows(IllegalStateException.class, () -> ids.getAnalysisId(file)))
        .hasMessageThat()
        .contains("Missing dense leaf ID for");
    assertThat(assertThrows(IllegalStateException.class, () -> ids.getAnalysisId(listing)))
        .hasMessageThat()
        .contains("Missing dense leaf ID for");
    assertThat(assertThrows(IllegalStateException.class, () -> ids.getSourceId(file)))
        .hasMessageThat()
        .contains("Missing dense leaf ID for source file");
    assertThat(ids.fallbackIdCount()).isEqualTo(0);
  }

  @Test
  public void dense_channelsAreIndependent() {
    var analysisOnly = new Object2IntOpenHashMap<FileOpNode>();
    analysisOnly.put(file, 0);
    var ids = LeafIdAssignment.dense(analysisOnly, new Object2IntOpenHashMap<>());
    assertThat(ids.getAnalysisId(file)).isEqualTo(0);
    assertThrows(IllegalStateException.class, () -> ids.getSourceId(file));

    var sourceOnly = new Object2IntOpenHashMap<FileKey>();
    sourceOnly.put(file, 0);
    var sourceIds = LeafIdAssignment.dense(new Object2IntOpenHashMap<>(), sourceOnly);
    assertThat(sourceIds.getSourceId(file)).isEqualTo(0);
    assertThrows(IllegalStateException.class, () -> sourceIds.getAnalysisId(file));
  }

  @Test
  public void lazy_assignsDisjointIdsAcrossChannels() {
    LeafIdAssignment ids = LeafIdAssignment.lazy();

    int fileId = ids.getAnalysisId(file);
    int listingId = ids.getAnalysisId(listing);
    int sourceId = ids.getSourceId(file);

    assertThat(ImmutableSet.of(fileId, listingId, sourceId)).hasSize(3);
    assertThat(ids.getAnalysisId(file)).isEqualTo(fileId);
    assertThat(ids.getAnalysisId(listing)).isEqualTo(listingId);
    assertThat(ids.getSourceId(file)).isEqualTo(sourceId);
    assertThat(ids.denseIdCount()).isEqualTo(0);
    assertThat(ids.fallbackIdCount()).isEqualTo(3);
  }

  private LeafIdAssignment dense(int fileId, int listingId, int sourceId) {
    var analysisIds = new Object2IntOpenHashMap<FileOpNode>();
    analysisIds.put(file, fileId);
    analysisIds.put(listing, listingId);
    var sourceIds = new Object2IntOpenHashMap<FileKey>();
    sourceIds.put(file, sourceId);
    return LeafIdAssignment.dense(analysisIds, sourceIds);
  }

  private RootedPath rootedPath(String path) {
    return RootedPath.toRootedPath(root, PathFragment.create(path));
  }
}

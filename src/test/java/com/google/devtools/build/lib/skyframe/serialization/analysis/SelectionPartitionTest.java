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
import static com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.EmptyFileOpNode.EMPTY_FILE_OP_NODE;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.AbstractNestedFileOpNodes;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.FileKey;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNodeOrEmpty;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.ArrayList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class SelectionPartitionTest {

  private final Root root =
      Root.fromPath(new InMemoryFileSystem(DigestHashFunction.SHA256).getPath("/root"));

  private final FileKey fileA = FileKey.create(rootedPath("a.txt"));
  private final FileKey fileB = FileKey.create(rootedPath("b.txt"));

  @Test
  public void uploadOrder_unrankedThenPostOrderThenOwned() {
    ActionLookupKey empty = key("empty");
    ActionLookupKey bareLeaf = key("bare_leaf");
    ActionLookupKey top = key("top");
    ActionLookupKey middle = key("middle");
    ActionLookupKey bottom = key("bottom");
    ActionLookupData action = ActionLookupData.create(top, 0);
    var nodes =
        ImmutableMap.<ActionLookupKey, FileOpNodeOrEmpty>builder()
            .put(empty, EMPTY_FILE_OP_NODE)
            .put(bareLeaf, fileA)
            .put(top, nested(/* postOrderIndex= */ 3))
            .put(middle, nested(/* postOrderIndex= */ 2))
            .put(bottom, nested(/* postOrderIndex= */ 1))
            .buildOrThrow();

    SelectionPartition partition =
        SelectionPartition.partition(
            ImmutableList.of(action, top, empty, middle, bareLeaf, bottom),
            key -> false,
            nodes::get);

    assertThat(partition.uploadOrder())
        .containsExactly(empty, bareLeaf, bottom, middle, top, action)
        .inOrder();
  }

  @Test
  public void uploadOrder_tiesKeepSelectionOrder() {
    ActionLookupKey first = key("first");
    ActionLookupKey second = key("second");
    ActionLookupKey third = key("third");
    AbstractNestedFileOpNodes shared = nested(/* postOrderIndex= */ 2);
    var nodes =
        ImmutableMap.<ActionLookupKey, FileOpNodeOrEmpty>of(
            first, shared, second, nested(/* postOrderIndex= */ 1), third, shared);

    SelectionPartition partition =
        SelectionPartition.partition(
            ImmutableList.of(first, second, third), key -> false, nodes::get);

    assertThat(partition.uploadOrder()).containsExactly(second, first, third).inOrder();
  }

  @Test
  public void partition_skippedKeysAreLeftOut() {
    ActionLookupKey kept = key("kept");
    ActionLookupKey skipped = key("skipped");
    AbstractNestedFileOpNodes keptNode = nested(/* postOrderIndex= */ 1);
    var nodes =
        ImmutableMap.<ActionLookupKey, FileOpNodeOrEmpty>of(
            kept, keptNode, skipped, nested(/* postOrderIndex= */ 1));

    SelectionPartition partition =
        SelectionPartition.partition(
            ImmutableList.of(skipped, kept, ActionLookupData.create(skipped, 0)),
            key -> key.equals(skipped) || key.equals(ActionLookupData.create(skipped, 0)),
            nodes::get);

    assertThat(partition.roots()).containsExactly(keptNode);
    assertThat(partition.uploadOrder()).containsExactly(kept);
  }

  @Test
  public void partition_ownedKeysContributeOwnerNodeToRoots_resolvingEachOwnerOnce() {
    ActionLookupKey owner = key("owner");
    AbstractNestedFileOpNodes ownerNode = nested(/* postOrderIndex= */ 1);
    var resolved = new ArrayList<ActionLookupKey>();
    var nodes = ImmutableMap.<ActionLookupKey, FileOpNodeOrEmpty>of(owner, ownerNode);

    SelectionPartition partition =
        SelectionPartition.partition(
            ImmutableList.of(
                ActionLookupData.create(owner, 0), ActionLookupData.create(owner, 1), owner),
            key -> false,
            key -> {
              resolved.add(key);
              return nodes.get(key);
            });

    assertThat(resolved).containsExactly(owner);
    // Duplicates are retained; the leaf walk dedups.
    assertThat(partition.roots()).containsExactly(ownerNode, ownerNode, ownerNode);
    assertThat(partition.uploadOrder())
        .containsExactly(
            owner, ActionLookupData.create(owner, 0), ActionLookupData.create(owner, 1))
        .inOrder();
  }

  @Test
  public void uploadOrder_unwalkedNode_throws() {
    ActionLookupKey unwalked = key("unwalked");
    var node =
        (AbstractNestedFileOpNodes) AbstractNestedFileOpNodes.from(ImmutableList.of(fileA, fileB));

    SelectionPartition partition =
        SelectionPartition.partition(ImmutableList.of(unwalked), key -> false, key -> node);

    assertThat(assertThrows(IllegalStateException.class, partition::uploadOrder))
        .hasMessageThat()
        .contains("the node was not reached by this walk");
  }

  /** A fresh nested node, finished at {@code postOrderIndex} as the leaf walk would. */
  private AbstractNestedFileOpNodes nested(int postOrderIndex) {
    var node =
        (AbstractNestedFileOpNodes) AbstractNestedFileOpNodes.from(ImmutableList.of(fileA, fileB));
    assertThat(node.markVisited()).isTrue();
    node.setPostOrderIndex(postOrderIndex);
    return node;
  }

  private static ActionLookupKey key(String name) {
    return ConfiguredTargetKey.builder()
        .setLabel(Label.parseCanonicalUnchecked("//pkg:" + name))
        .build();
  }

  private RootedPath rootedPath(String path) {
    return RootedPath.toRootedPath(root, PathFragment.create(path));
  }
}

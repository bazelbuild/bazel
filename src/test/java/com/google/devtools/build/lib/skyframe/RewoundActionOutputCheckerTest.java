// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileContentsProxy;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class RewoundActionOutputCheckerTest {
  private static final FileArtifactValue VALUE_1_REMOTE =
      FileArtifactValue.createForRemoteFile(
          /* digest= */ new byte[0], /* size= */ 0, /* locationIndex= */ 1);
  private static final FileArtifactValue VALUE_2_REMOTE =
      FileArtifactValue.createForRemoteFile(
          /* digest= */ new byte[0], /* size= */ 0, /* locationIndex= */ 2);

  private static final ActionLookupKey KEY = ActionsTestUtil.NULL_ARTIFACT_OWNER;
  private static final ActionLookupData ACTION_LOOKUP_DATA_1 = ActionLookupData.create(KEY, 1);

  private static final ArtifactRoot OUTPUT_ROOT =
      ArtifactRoot.asDerivedRoot(new Scratch().resolve("/execroot"), RootType.OUTPUT, "out");

  @Test
  public void regeneratedContent_ignoresLocationAndProxy() {
    Artifact original = output("file", ACTION_LOOKUP_DATA_1);
    Artifact regenerated = output("file", ACTION_LOOKUP_DATA_1);
    var before = createWithArtifactData(ImmutableMap.of(original, VALUE_1_REMOTE));
    var after =
        createWithArtifactData(
            ImmutableMap.of(
                regenerated,
                FileArtifactValue.createForNormalFile(
                    new byte[0], new FileContentsProxy(1, 2, 3), 0)));

    assertThat(changedOutput(after, before)).isNull();
  }

  @Test
  public void regeneratedContent_detectsChangedSecondaryOutput() {
    Artifact primary = output("primary");
    Artifact secondary = output("secondary");
    var before =
        createWithArtifactData(ImmutableMap.of(primary, VALUE_1_REMOTE, secondary, VALUE_1_REMOTE));
    var after =
        createWithArtifactData(
            ImmutableMap.of(
                primary,
                VALUE_2_REMOTE,
                secondary,
                FileArtifactValue.createForRemoteFile(new byte[] {1}, 0, 1)));

    assertThat(changedOutput(after, before)).isEqualTo(secondary.getExecPathString());
  }

  @Test
  public void regeneratedContent_detectsChangedSizeAndType() {
    Artifact file = output("file");
    var before = createWithArtifactData(ImmutableMap.of(file, VALUE_1_REMOTE));
    for (FileArtifactValue changed :
        ImmutableList.of(
            FileArtifactValue.createForRemoteFile(new byte[0], 1, 1),
            FileArtifactValue.createForDirectoryWithHash(new byte[0]))) {
      assertThat(changedOutput(createWithArtifactData(ImmutableMap.of(file, changed)), before))
          .isEqualTo(file.getExecPathString());
    }
  }

  @Test
  public void regeneratedContent_detectsOutputMissingFromPreviousValue() {
    Artifact first = output("first");
    Artifact second = output("second");
    var one = createWithArtifactData(ImmutableMap.of(first, VALUE_1_REMOTE));
    var two =
        createWithArtifactData(ImmutableMap.of(first, VALUE_1_REMOTE, second, VALUE_1_REMOTE));

    assertThat(changedOutput(two, one)).isEqualTo(second.getExecPathString());
  }

  @Test
  public void regeneratedContent_comparesTreeChildrenByContent() {
    TreeArtifact tree = tree("tree");
    TreeFileArtifact child = TreeFileArtifact.createTreeOutput(tree, "child");
    var before =
        createWithTreeArtifactData(
            ImmutableMap.of(
                tree, TreeArtifactValue.newBuilder(tree).putChild(child, VALUE_1_REMOTE).build()));
    var materialized =
        createWithTreeArtifactData(
            ImmutableMap.of(
                tree,
                TreeArtifactValue.newBuilder(tree)
                    .putChild(child, FileArtifactValue.createForNormalFile(new byte[0], null, 0))
                    .build()));
    var changed =
        createWithTreeArtifactData(
            ImmutableMap.of(
                tree,
                TreeArtifactValue.newBuilder(tree)
                    .putChild(child, FileArtifactValue.createForRemoteFile(new byte[] {1}, 0, 1))
                    .build()));
    var empty = createWithTreeArtifactData(ImmutableMap.of(tree, TreeArtifactValue.empty()));

    assertThat(changedOutput(materialized, before)).isNull();
    assertThat(changedOutput(changed, before)).isEqualTo(child.getExecPathString());
    assertThat(changedOutput(empty, before)).isEqualTo(child.getExecPathString());
    assertThat(changedOutput(before, empty)).isEqualTo(child.getExecPathString());
  }

  @Test
  public void regeneratedContent_distinguishesEmptyTreeFromMissingMetadata() {
    TreeArtifact tree = tree("tree");
    var empty = createWithTreeArtifactData(ImmutableMap.of(tree, TreeArtifactValue.empty()));
    var missingMetadata = createWithArtifactData(ImmutableMap.of(output("other"), VALUE_1_REMOTE));

    TreeArtifactValue regenerated = TreeArtifactValue.empty();

    assertThat(
            RewoundActionOutputChecker.getChangedOutput(
                tree, missingMetadata, regenerated.getMetadata(), regenerated))
        .isEqualTo(tree.getExecPathString());
    assertThat(
            RewoundActionOutputChecker.getChangedOutput(
                tree, empty, regenerated.getMetadata(), regenerated))
        .isNull();
  }

  @Test
  public void regeneratedContent_ignoresTreeArchive() {
    TreeArtifact tree = tree("tree");
    TreeFileArtifact child = TreeFileArtifact.createTreeOutput(tree, "child");
    var before =
        createWithTreeArtifactData(
            ImmutableMap.of(
                tree, TreeArtifactValue.newBuilder(tree).putChild(child, VALUE_1_REMOTE).build()));
    var after =
        TreeArtifactValue.newBuilder(tree)
            .putChild(child, VALUE_2_REMOTE)
            .setArchivedRepresentation(
                ArchivedTreeArtifact.createForTree(tree),
                FileArtifactValue.createForRemoteFile(new byte[] {2}, 100, 1))
            .build();

    assertThat(
            RewoundActionOutputChecker.getChangedOutput(tree, before, after.getMetadata(), after))
        .isNull();
  }

  @Test
  public void regeneratedContent_detectsChangedSymlinkTarget() throws Exception {
    Artifact symlink = output("symlink");
    var path = symlink.getPath();
    path.getParentDirectory().createDirectoryAndParents();
    path.createSymbolicLink(PathFragment.create("first"));
    var before =
        createWithArtifactData(
            ImmutableMap.of(symlink, FileArtifactValue.createForUnresolvedSymlink(path)));
    path.delete();
    path.createSymbolicLink(PathFragment.create("second"));
    var after =
        createWithArtifactData(
            ImmutableMap.of(symlink, FileArtifactValue.createForUnresolvedSymlink(path)));

    assertThat(changedOutput(after, before)).isEqualTo(symlink.getExecPathString());
  }

  /**
   * Checks every output of the regenerated action, like {@code SkyframeActionExecutor} does. A
   * regenerated action always produces metadata for all of its declared outputs.
   */
  private static String changedOutput(ActionExecutionValue after, ActionExecutionValue before) {
    for (Artifact output :
        ImmutableSet.<Artifact>builder()
            .addAll(after.getAllFileValues().keySet())
            .addAll(after.getAllTreeArtifactValues().keySet())
            .build()) {
      TreeArtifactValue tree = after.getAllTreeArtifactValues().get(output);
      String changed =
          RewoundActionOutputChecker.getChangedOutput(
              output,
              before,
              tree != null ? tree.getMetadata() : after.getAllFileValues().get(output),
              tree);
      if (changed != null) {
        return changed;
      }
    }
    return null;
  }

  private static ActionExecutionValue createWithArtifactData(
      ImmutableMap<Artifact, FileArtifactValue> artifactData) {
    return ActionExecutionValue.create(
        /* artifactData= */ artifactData,
        /* treeArtifactData= */ ImmutableMap.of(),
        /* richArtifactData= */ null,
        /* discoveredModules= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER));
  }

  private static ActionExecutionValue createWithTreeArtifactData(
      ImmutableMap<Artifact, TreeArtifactValue> treeArtifactData) {
    return ActionExecutionValue.create(
        /* artifactData= */ ImmutableMap.of(),
        treeArtifactData,
        /* richArtifactData= */ null,
        /* discoveredModules= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER));
  }

  private static DerivedArtifact output(String rootRelativePath) {
    return output(rootRelativePath, ACTION_LOOKUP_DATA_1);
  }

  private static DerivedArtifact output(
      String rootRelativePath, ActionLookupData generatingAction) {
    DerivedArtifact result =
        DerivedArtifact.create(
            OUTPUT_ROOT,
            OUTPUT_ROOT.getExecPath().getRelative(rootRelativePath),
            generatingAction.getActionLookupKey());
    result.setGeneratingActionKey(generatingAction);
    return result;
  }

  private static TreeArtifact tree(String rootRelativePath) {
    TreeArtifact result =
        TreeArtifact.create(
            OUTPUT_ROOT, OUTPUT_ROOT.getExecPath().getRelative(rootRelativePath), KEY);
    result.setGeneratingActionKey(ACTION_LOOKUP_DATA_1);
    return result;
  }
}
